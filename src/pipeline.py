import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List

# --- Configuração de Ambiente ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOG_LEVEL'] = '3'
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    module='tensorflow_addons'
)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# --- Fim da Configuração ---

import matplotlib
matplotlib.use('Agg')  # Usa backend não-interativo (antes de importar pyplot)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import pandas as pd
import numpy as np

# Imports dos seus módulos
from data.data_preparation import (
    filter_data_for_iteration, prepare_train_test_sets, initialize_prediction_lists,
    calculate_total_iterations, organize_wells_by_df_size, apply_custom_kalman_filter
)
from evaluation.evaluation import evaluate_and_plot_all_wells
from training.train_utils import fine_tune_and_predict_well, prepare_args_for_fine_tuning
from training.models_forecast import train_and_evaluate_disruptive
from utils.utilities import apply_filter_to_predictions, print_style
from data.data_loading import DataSource

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import shutil

def _suffix_path(base: str | Path, suffix: str) -> Path:
    p = Path(base)
    return p.with_stem(f"{p.stem}_{suffix}")

def _downsample(df: pd.DataFrame | dict, step: int, wells: list):
    if isinstance(df, dict):
        return [df[w].iloc[::step] for w in wells]
    return df.iloc[::step]


# ==============================================================================
# CLASSE DE PIPELINE REATORADA PARA ENSEMBLES
# ==============================================================================
class WellForecastPipeline:
    def __init__(
        self,
        dataset: str,
        wells: List[str],
        serie_name: str,
        data_path: str,
        forecast_steps: int,
        window_size: int,
        train_windows: int,
        fine_tuning_windows: int,
        model_type: str,
        architecture_name: str = "Generic",
        sample_time: int = 1,
        model_path: str | Path = "best_model.keras",
        cum_sum: bool = False,
        data_loader_kwargs: Dict[str, Any] | None = None,
        variable_mapping: Dict[str, str] | None = None,
        filter_postprocess: Callable | None = None,
        model_creation_kwargs: Dict[str, Any] = None, # ★ NOVO PARÂMETRO
        n_ensembles: int = 1,
    ) -> None:
        # --- Parâmetros de Configuração ---
        self.dataset = dataset
        self.wells = wells
        self.serie_name = (variable_mapping.get(serie_name) if variable_mapping else serie_name)
        self.data_path = data_path
        self.forecast_steps = forecast_steps
        self.window_size = window_size
        self.train_windows = train_windows
        self.fine_tuning_windows = fine_tuning_windows
        self.model_type = model_type
        self.architecture_name = architecture_name
        self.sample_time = sample_time
        self.model_path = Path(model_path)
        self.cum_sum = cum_sum
        self.data_loader_kwargs = data_loader_kwargs or {}
        self.variable_mapping = variable_mapping
        self.filter_fn = filter_postprocess or apply_custom_kalman_filter
        self.model_creation_kwargs = model_creation_kwargs or {} # ★ ARMAZENA OS KWARGS
        self.n_ensembles = n_ensembles

        # --- Estado do Pipeline ---
        self.df_list: List[pd.DataFrame] = []
        self.total_iters: int = 0
        self.y_test_truth: List[List[float]] = [] # Armazena o y_test de referência

    def run(self):
        """
        Executa o pipeline de ensemble, garantindo o isolamento de cada membro.
        """
        # 1. Carrega os dados uma única vez para consistência e eficiência.
        self._load_and_prepare_data()
        
        # Lista para armazenar as predições de cada membro do ensemble.
        all_ensemble_preds: List[List[List[float]]] = []

        # 2. Executa um pipeline completo para cada membro do ensemble.
        for k in range(self.n_ensembles):
            print(f"\n{'='*20} [ ENSEMBLE MEMBER {k + 1}/{self.n_ensembles} ] {'='*20}")

            # Isola o caminho do modelo para este membro específico.
            member_model_path = self.model_path.with_stem(f"{self.model_path.stem}_ensemble_{k}")
            
            # Executa o pipeline para este membro e obtém suas predições.
            y_pred_member, y_test_member = self._run_single_pipeline_member(member_model_path, seed=k)
            
            # Acumula as predições do membro.
            all_ensemble_preds.append(y_pred_member)

            # Armazena os valores de teste do primeiro membro como a "verdade" de referência.
            if k == 0:
                self.y_test_truth = y_test_member

        # 3. Agrega os resultados e realiza a avaliação final.
        self._final_eval_ensemble(all_ensemble_preds)

    def _load_and_prepare_data(self):
        """Carrega e pré-processa os dados uma única vez."""
        print("[Setup] Carregando e preparando dados...")
        cfg = {"name": self.dataset, "wells": self.wells, "serie_name": self.serie_name,
               "load_params": {**self.data_loader_kwargs, "data_path": self.data_path, "cum_sum": self.cum_sum},
               "variable_mapping": self.variable_mapping,
               "features": ["BORE_GAS_VOL", "CE", "delta_P", "PI", "AVG_DOWNHOLE_PRESSURE", "BORE_WAT_VOL",
                            "ON_STREAM_HRS", "Tempo_Inicio_Prod", "Taxa_Declinio", "BORE_OIL_VOL"]}
        raw = DataSource(cfg).get_loader().load()
        self.df_list = _downsample(raw, self.sample_time, self.wells)
        self.wells, self.df_list = organize_wells_by_df_size(self.wells, self.df_list)
        self.total_iters = calculate_total_iterations(self.df_list)
        print(f"[Data] {len(self.wells)} poços carregados. {self.total_iters} iterações por execução.")

    def _run_single_pipeline_member(self, member_model_path: Path, seed: int) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Executa um ciclo completo para um único membro, retornando suas predições e dados de teste.
        """
        # Define sementes aleatórias para diversidade e reprodutibilidade.
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Inicializa o estado para este membro.
        y_test_list, y_pred_list = initialize_prediction_lists(len(self.wells))
        active_wells = list(range(len(self.wells)))

        for it in range(self.total_iters):
            if it > 0 and it % 1000 == 0:
                print(f"  [Iteração {it}/{self.total_iters}]")

            sets = filter_data_for_iteration(self.df_list, self.window_size, self.serie_name, self.forecast_steps,
                                             it, active_wells, self.train_windows, self.fine_tuning_windows,
                                             cum_sum=self.cum_sum)
            if not sets: continue

            X_tr, y_tr, X_ts, y_ts, max_tr, scalers = prepare_train_test_sets(sets, self.model_type, well_idx=0)
            
            # Treina o modelo base (se for a primeira iteração) em um caminho isolado.
            if self.model_type == "DL" and it == 0:
                self._train_base_model_for_member(sets, active_wells, member_model_path)
                continue

            args, active_wells = prepare_args_for_fine_tuning(sets, X_ts, y_ts, max_tr, scalers,
                                                              self.model_type, str(member_model_path),
                                                              self.wells, active_wells, self.cum_sum, it)
            if not args: continue

            for arg in args:
                result = fine_tune_and_predict_well(arg)
                if result:
                    idx, y_true_step, y_pred_step = result
                    y_test_list[idx].extend(y_true_step)
                    y_pred_list[idx].extend(y_pred_step)
        
        return y_pred_list, y_test_list

    def _train_base_model_for_member(self, sets, active_wells, member_model_path: Path):
        """Treina o modelo base para um membro, garantindo isolamento total."""
        print(f"  [Modelo Base] Treinando para o membro com caminho base: {member_model_path}")
        
        for i in active_wells:
            X_b, y_b, *_ = prepare_train_test_sets(sets, self.model_type, well_idx=i)
            # Cria um caminho único para o modelo deste poço e deste membro do ensemble.
            path = _suffix_path(member_model_path, self.wells[i].replace("/", "_"))
            
            # Força o retreinamento do zero, garantindo que não haja vazamento de estado.
            if path.exists():
                path.unlink()

            train_and_evaluate_disruptive(X_b, y_b, model_path=str(path), fine_tune=False,
                                          architecture_name=self.architecture_name,
                                          model_creation_kwargs=self.model_creation_kwargs)

    def _final_eval_ensemble(self, all_ensemble_preds: List[List[List[float]]]):
        """Agrega as predições do ensemble e avalia o resultado final."""
        print(f"\n{'='*20} [ AVALIAÇÃO FINAL DO ENSEMBLE ] {'='*20}")
        if not self.y_test_truth:
            print("AVISO: Nenhum dado de teste de referência encontrado para avaliação.")
            return

        num_members = len(all_ensemble_preds)
        num_wells = len(self.wells)
        aggregated_preds = [[] for _ in range(num_wells)]

        for i in range(num_wells):
            # Coleta as predições de todos os membros para este poço.
            preds_for_well = [all_ensemble_preds[k][i] for k in range(num_members) if i < len(all_ensemble_preds[k])]
            if not preds_for_well: continue

            # Garante que todas as listas de predição e a lista de teste tenham o mesmo comprimento.
            try:
                min_len = min(len(p) for p in preds_for_well)
                min_len = min(min_len, len(self.y_test_truth[i]))
                
                # Trunca todas as listas para o comprimento mínimo para um alinhamento perfeito.
                truncated_preds = [p[:min_len] for p in preds_for_well]
                self.y_test_truth[i] = self.y_test_truth[i][:min_len]
                
                if truncated_preds:
                    # Calcula a média ponto a ponto das predições do ensemble.
                    aggregated_preds[i] = np.mean(np.array(truncated_preds), axis=0).tolist()
            except (ValueError, IndexError) as e:
                print(f"AVISO: Falha ao agregar predições para o poço {i}: {e}")
                continue

        y_pred_filt = apply_filter_to_predictions(aggregated_preds, self.filter_fn)
        for tag, preds in {"Kalman": y_pred_filt, "No Filter": aggregated_preds}.items():
            print_style(tag)
            evaluate_and_plot_all_wells(self.dataset, self.wells, self.y_test_truth, preds,
                                        self.window_size, self.forecast_steps,
                                        metrics_accumulator=[], method=tag)

