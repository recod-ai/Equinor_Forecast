import numpy as np
import os
import tensorflow as tf
tf.keras.backend.clear_session()  # Clear the session
from tensorflow.keras.models import load_model
from training.models_forecast import train_and_evaluate_disruptive, train_and_evaluate_XGB

import time
from typing import Any, List, Dict, Callable, Tuple, Union
import math
from pathlib import Path
import logging

from prediction.prediction_utils import make_predictions_for_all_wells, inverse_transform_predictions

def insert_suffix_before_extension(base_path, suffix):
    base_path = Path(base_path)
    return base_path.with_stem(f"{base_path.stem}_{suffix}")


###############################################################################
# Função para preparar argumentos para fine tuning
###############################################################################
def prepare_args_for_fine_tuning(
    sets: List[Any],
    X_tests: List[Any],
    y_tests: List[Any],
    max_trains: List[Any],
    scalers: List[Any],
    model_type: str,
    model_path: str,
    wells: List[str],
    active_wells: List[int],
    cum_sum: bool,
    control_iteration: int,
) -> Tuple[List[tuple], List[int]]:
    """
    Prepara os argumentos para ajuste fino e predição para cada poço.

    Retorna:
        args_list: Lista de argumentos para fine tuning e predição.
        active_wells: Lista atualizada de índices de poços ativos.
    """
    args_list = []
    wells_to_remove = []

    for idx, i in enumerate(active_wells):
        if idx >= len(sets) or sets[idx] is None:
            print(f"Sem dados restantes para o poço {wells[i]} na iteração {control_iteration + 1}. Removendo dos poços ativos.")
            wells_to_remove.append(i)
            continue

        sets_i = sets[idx]
        X_tests_i = X_tests[idx]
        y_tests_i = y_tests[idx]
        scalers_i = scalers[idx]
        max_train_i = max_trains[idx]

        well_model_path = str(insert_suffix_before_extension(model_path, wells[i].replace("/", "_")))

        args_list.append((
            i, wells[i], sets_i, X_tests_i, y_tests_i, max_train_i, scalers_i,
            model_type, well_model_path, True, cum_sum, control_iteration
        ))
            

    # Atualiza a lista de poços ativos removendo os índices sem dados
    active_wells = [i for i in active_wells if i not in wells_to_remove]
    return args_list, active_wells

# ─────────────────────────────────────────────────────────────────────────────
# Fine‑tune worker – warm‑start models kept alive inside worker process
# ─────────────────────────────────────────────────────────────────────────────

# src/analysis/shap_analyzer.py

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial

def _prediction_wrapper(data, model, original_shape):
    """
    Internal wrapper function to bridge SHAP's 2D data format with the model's 3D input.
    
    Args:
        data: Data provided by SHAP explainer (can be a numpy array or DenseData object).
        model: The Keras model to use for prediction.
        original_shape (tuple): The original 3D shape of the input data (samples, timesteps, features).
    
    Returns:
        np.ndarray: A 1D array of model predictions.
    """
    # Ensure data is a numpy array
    data_np = np.array(data)
    
    # Reshape to the 3D format the model expects
    time_steps, num_features = original_shape[1], original_shape[2]
    data_3d = data_np.reshape(-1, time_steps, num_features)
    
    # Predict and flatten the output to a 1D vector
    predictions = model.predict(data_3d, verbose=0)
    return predictions.flatten()

from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import warnings

def choose_k_by_silhouette(
    X: np.ndarray,
    k_grid=(10, 25, 50, 75, 100),
    sample_size: int = 20_000,
    random_state: int = 0,
    threshold: float = 0.98
) -> int:
    """Return the smallest k whose silhouette ≥ threshold·max_score."""
    if len(X) > sample_size:
        rng = np.random.default_rng(random_state)
        X_sample = X[rng.choice(len(X), sample_size, replace=False)]
    else:
        X_sample = X

    scores = {}
    for k in k_grid:
        km = MiniBatchKMeans(k, random_state=random_state).fit(X_sample)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            s = silhouette_score(X_sample, km.labels_)
        scores[k] = s

    max_s = max(scores.values())
    for k in sorted(k_grid):
        if scores[k] >= threshold * max_s:
            return k
    return max(scores, key=scores.get)  # fallback


def run_shap_analysis(
    model,
    X_data,
    feature_names,
    iteration,
    well_name,
    output_dir: Path
):
    """
    Performs a full, on-the-fly SHAP analysis for a live Keras model and saves the results.

    Args:
        model: The "live" Keras model object from the training loop.
        X_data (np.ndarray): The input data for this iteration (e.g., X_train), in 3D format.
        feature_names (list): A list of feature names.
        iteration (int): The current training iteration number.
        well_name (str): The name of the well being analyzed.
        output_dir (Path): The base directory to save the analysis artifacts.
    """
    print(f"\n--- [SHAP Analysis] Starting for Iteration {iteration} - Well {well_name} ---")

    # --- 1. Model Sanity Check ---
    X_data = X_data[:]
    print("[SHAP Sanity Check] Verifying model sensitivity...")
    if len(X_data) < 2:
        print("[SHAP WARNING] Not enough data for a meaningful sanity check. Skipping.")
    else:
        test_samples = X_data[:2].copy() # Using just 2 samples is enough
        original_preds = model.predict(test_samples, verbose=0).flatten()
        
        test_samples[:, 0, 0] *= 1.5 # Perturb the first feature of the first timestep
        perturbed_preds = model.predict(test_samples, verbose=0).flatten()
        
        difference = np.sum(np.abs(original_preds - perturbed_preds))
        if difference < 1e-6:
            print(f"[SHAP WARNING] Model appears insensitive to feature changes (difference: {difference:.2e}). SHAP values might be zero.")
            # We continue the analysis财富, but this warning is crucial.
        else:
            print("[SHAP Sanity Check] ✅ Model is sensitive. Proceeding with analysis.")

    # --- 2. Prepare Data for KernelExplainer ---
    if len(X_data) == 0:
        print("[SHAP ERROR] X_data is empty. Aborting analysis for this iteration.")
        return

    num_samples, _, _ = X_data.shape
    X_data_2d = X_data.reshape(num_samples, -1)

    # Use a K-Means summary of the data as the background reference for SHAP
    # Limiting to 50 background points is a good trade-off for speed and accuracy
    
    k = choose_k_by_silhouette(X_data_2d)
    print("✅ BEST K: ", k)
    background_data_2d = shap.kmeans(X_data_2d, k)

    # Use all available data points for the explanation to get a comprehensive view
    explain_data_2d = X_data_2d[:5]
    
    # --- 3. Compute SHAP Values ---
    print(f"Computing SHAP values with KernelExplainer for {len(explain_data_2d)} samples...")
    
    # Create a partial function to pass the model and original shape to the wrapper
    bound_prediction_wrapper = partial(_prediction_wrapper, model=model, original_shape=X_data.shape)
    
    explainer = shap.KernelExplainer(bound_prediction_wrapper, background_data_2d)
    
    # 'nsamples="auto"' lets SHAP choose a reasonable number of perturbations.
    shap_values = explainer.shap_values(explain_data_2d, nsamples='auto')

    # --- 4. Generate and Save Visualizations ---
    well_name_safe = well_name.replace('/', '_').replace(' ', '_')
    iter_output_dir = output_dir / f"iter_{iteration}"
    iter_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving results to: {iter_output_dir}")

    # Plot 1: Beeswarm Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, explain_data_2d, feature_names=feature_names, show=False)
    plt.title(f"SHAP Value Summary - Well {well_name} - Iteration {iteration}", fontsize=14)
    plt.tight_layout()
    plt.savefig(iter_output_dir / "summary_plot_beeswarm.png")
    plt.close()

    # Plot 2: Bar Plot (Mean Absolute Importance)
    plt.figure()
    shap.summary_plot(shap_values, explain_data_2d, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"Feature Importance - Well {well_name} - Iteration {iteration}", fontsize=14)
    plt.tight_layout()
    plt.savefig(iter_output_dir / "feature_importance_bar.png")
    plt.close()
    
    # --- 5. Save Raw SHAP Data ---
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs_shap})
    shap_df.to_csv(iter_output_dir / "feature_importance_values.csv", index=False)

    print(f"--- [SHAP Analysis] Analysis for Iteration {iteration} complete. ---")

# ----------------------------------------------------------------------
# Helper: choose a compact background with silhouette-based K-means
# ----------------------------------------------------------------------
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np
import warnings
from pathlib import Path
import shap, tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Tuple, Optional # Adicionei imports faltantes para o exemplo funcionar

# Supondo que a função choose_k_by_silhouette exista em outro lugar
# from your_utils import choose_k_by_silhouette
# Criando uma função dummy para o código rodar:
def choose_k_by_silhouette(X_flat, k_grid):
    print("Aviso: Usando função dummy choose_k_by_silhouette. Retornando k=10.")
    return 10

# ----------------------------------------------------------------------
# 1. Função make_vector_output_model (mantida como está)
# ----------------------------------------------------------------------
def make_vector_output_model(
    model: tf.keras.Model,
    reduction: str = "last",
):
    """
    Retorna uma *view* do `model` cuja saída é um vetor adequado para o SHAP.
    """
    if reduction == "last":
        out = model.output[:, -1, ...]
    elif reduction == "mean":
        out = tf.reduce_mean(model.output, axis=1)
    else:
        raise ValueError("reduction must be 'last' or 'mean'")
    return tf.keras.Model(inputs=model.input, outputs=out)

# ----------------------------------------------------------------------
# 2. Robust SHAP → NumPy extractor (mantida como está)
# ----------------------------------------------------------------------
def get_shap_array(obj: Any) -> np.ndarray:
    """
    Retorna um array NumPy de qualquer tipo de retorno do SHAP.
    Funciona com shap.Explanation (0.44+), list, tuple, ndarray, tensor.
    """
    if hasattr(obj, "values"):
        return np.asarray(obj.values)
    if isinstance(obj, (list, tuple)):
        return np.asarray(obj[0])
    return np.asarray(obj)

import json, textwrap, pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from typing import List, Optional


def generate_iteration_report(
    shap_values: np.ndarray,
    X_data_2d: np.ndarray,
    feature_names: List[str],
    iteration: int,
    output_dir: Path,
    prev_rank: Optional[pd.Series] = None,
) -> pd.Series:
    ...
    abs_shap   = np.abs(shap_values)

    # ---------- 1. Global metrics ----------
    mean_abs_vec = abs_shap.mean(axis=0)                         # 1-D array
    mean_abs     = pd.Series(mean_abs_vec, index=feature_names)  # ← NEW
    total        = mean_abs.sum()
    rel_import   = mean_abs / total                              # Series
    rank         = rel_import.rank(ascending=False,
                                   method="min").astype(int)
    gini_coeff = 1 - 2 * np.trapz(
        np.sort(rel_import), dx=1/len(rel_import)
    )  # 0 = equal, 1 = totally concentrated

    # ---------- 2. Directionality ----------
    dir_sign = []
    for i, f in enumerate(feature_names):
        rho, _ = spearmanr(X_data_2d[:, i], shap_values[:, i])
        dir_sign.append("↑" if rho > 0 else "↓")

    # ---------- 3. Convergence ----------
    conv_metric = None
    if prev_rank is not None:
        conv_metric, _ = spearmanr(
            prev_rank.loc[feature_names], rank
        )

    # ---------- 4. Compact table ----------
    table = pd.DataFrame({
        "rank": rank,
        "feature": feature_names,
        "mean_|SHAP|": mean_abs,
        "relative_%": rel_import.mul(100),
        "direction": dir_sign,
    }).sort_values("rank")

    # ---------- 5. Save artifacts ----------
    csv_path  = output_dir / f"iter_{iteration:04}_summary.csv"
    md_path   = output_dir / f"iter_{iteration:04}_report.md"
    json_path = output_dir / f"iter_{iteration:04}_metrics.json"
    table.to_csv(csv_path, index=False)

    # basic JSON for programmatic inspection
    json.dump(
        {
            "iteration": iteration,
            "gini": float(gini_coeff),
            "spearman_vs_prev": None if conv_metric is None else float(conv_metric),
            "top_feature": table.iloc[0]["feature"],
        },
        open(json_path, "w"), indent=2
    )

    # Markdown
    md = textwrap.dedent(f"""
    # SHAP Report – Iteration {iteration}

    **Top 10 features**

    | Rank | Feature | Mean |SHAP| | Share (%) | Direction |
    |------|---------|-------------|-----------|-----------|
    {table.head(10).to_markdown(index=False)}        # ← headers defaults to "keys"

    *Total importance is normalised to 100 %. Direction = monotonic sign of
    the SHAP dependence (Spearman).*  

    **Gini coefficient** of importance distribution: **{gini_coeff:0.3f}**  
    """)
    if conv_metric is not None:
        md += f"**Spearman correlation with previous iteration**: **{conv_metric:0.3f}**\n"

    md_path.write_text(md.strip())

    print(f"📄 Saved CSV → {csv_path.name}, Markdown → {md_path.name}, JSON → {json_path.name}")
    return rank

# ----------------------------------------------------------------------
# 3. Versão final e corrigida de run_shap_gradient_analysis
# ----------------------------------------------------------------------
def run_shap_gradient_analysis(
    model: tf.keras.Model,
    X_data: np.ndarray,
    feature_names: list,
    iteration: int,
    well_name: str,
    output_dir: Path,
    k_grid: Tuple[int, ...] = (10, 25, 50, 75, 100),
    output_reduction: str = "last",
):
    """
    Análise SHAP baseada em gradiente para modelos de sequência, agora corrigida e robusta.
    """
    print(f"\n--- [Gradient SHAP] Iter {iteration} – Well {well_name} ---")

    # ──────────────────────────────────────────────────────────────────
    # 0 ▸ Envolve o modelo para que sua saída seja um vetor (limitação do SHAP)
    # ──────────────────────────────────────────────────────────────────
    # Removida a definição duplicada da função. Usando a global.
    shap_ready_model = make_vector_output_model(model, output_reduction)

    # ──────────────────────────────────────────────────────────────────
    # 1 ▸ Escolhe um background compacto via K-means com silhueta
    # ──────────────────────────────────────────────────────────────────
    n_samples, time_steps, n_feats = X_data.shape
    X_flat = X_data.reshape(n_samples, -1)

    k = choose_k_by_silhouette(X_flat, k_grid=k_grid)
    print(f"[BG] Silhouette-optimised k = {k}")

    centers_flat = shap.kmeans(X_flat, k)
    # Compatibilidade com versões antigas/novas do shap.kmeans
    background_kmeans = getattr(centers_flat, "data", np.asarray(centers_flat))
    background_3d = background_kmeans.reshape(k, time_steps, n_feats)

    # ──────────────────────────────────────────────────────────────────
    # 2 ▸ Computa o SHAP de gradiente e CORRIGE O SHAPE
    # ──────────────────────────────────────────────────────────────────
    explainer = shap.GradientExplainer(shap_ready_model, background_3d)
    shap_raw = explainer(X_data)
    shap_vals_raw = get_shap_array(shap_raw) # Saída pode ser 4D: (samples, time, feats, outputs)

    # ▼▼▼ INÍCIO DA CORREÇÃO ▼▼▼

    # O GradientExplainer retorna um SHAP value por neurônio de saída.
    # Como nosso modelo tem 1 saída, a forma é (n_samples, time_steps, n_feats, 1).
    # Precisamos remover essa última dimensão para obter (n_samples, time_steps, n_feats).
    if shap_vals_raw.ndim == 4 and shap_vals_raw.shape[-1] == 1:
        print(f"Formato SHAP original: {shap_vals_raw.shape}. Removendo a última dimensão.")
        shap_vals_3d = np.squeeze(shap_vals_raw, axis=-1)
    else:
        # Mantém a lógica de fallback se a forma for diferente do esperado
        shap_vals_3d = shap_vals_raw

    # Validação final do formato
    expected_shape = (n_samples, time_steps, n_feats)
    if shap_vals_3d.shape != expected_shape:
        raise ValueError(
            f"Formato SHAP final {shap_vals_3d.shape} é inesperado. "
            f"Esperava-se {expected_shape}."
        )

    # Agora `shap_vals_3d` tem o formato 3D correto.
    # O reshape antigo não é mais necessário.
    mean_abs_shap = np.abs(shap_vals_3d).mean(axis=(0, 1)) # Média sobre amostras e tempo

    # Para o gráfico beeswarm, precisamos de um formato 2D (samples, features_flat)
    shap_vals_2d = shap_vals_3d.reshape(n_samples, -1)

    # ──────────────────────────────────────────────────────────────────
    # 3 ▸ Salva os artefatos
    # ──────────────────────────────────────────────────────────────────
    iter_dir = output_dir / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
    ).to_csv(iter_dir / "feature_importance_values.csv", index=False)

    # Gráfico Beeswarm (pontos por T×F) •••
    shap.summary_plot(
        shap_vals_2d, # Usa a versão 2D achatada
        X_flat,
        feature_names=[f"{f}" for t in range(time_steps) for f in feature_names],
        show=False, plot_type="dot"
    )
    plt.title(f"Gradient SHAP Beeswarm – {well_name} – Iter {iteration}")
    plt.tight_layout()
    plt.savefig(iter_dir / "summary_plot_beeswarm.png", dpi=300)
    plt.close()

    # Gráfico de Barras (importância global por feature) •••
    shap.summary_plot(
        mean_abs_shap.reshape(1, -1),       # ← reshape to (1, n_feats)
        features=np.zeros((1, n_feats)), 
        feature_names=feature_names,
        show=False, plot_type="bar"
    )
    plt.title(f"Gradient SHAP Importance – {well_name} – Iter {iteration}")
    plt.tight_layout()
    plt.savefig(iter_dir / "feature_importance_bar.png", dpi=300)
    plt.close()

    rank_series = generate_iteration_report(
        shap_values=shap_vals_2d,           # (samples, T*F)
        X_data_2d=X_flat,
        feature_names=[f"{f}_t{t}" for t in range(time_steps) for f in feature_names],
        iteration=iteration,
        output_dir=iter_dir,
        prev_rank=run_shap_gradient_analysis.prev_rank
            if hasattr(run_shap_gradient_analysis, "prev_rank") else None
    )

    print(f"✓ Artefatos SHAP salvos em {iter_dir.resolve()}")





# src/analysis/sensitivity_analyzer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def run_sensitivity_analysis(
    model,
    X_data, # Usaremos X_train ou X_test aqui
    feature_names,
    iteration,
    well_name,
    output_dir: Path
):
    """
    Executa uma análise de sensibilidade manual (feature perturbation) e salva os resultados.

    Args:
        model: O modelo Keras "vivo" em memória.
        X_data (np.ndarray): Um conjunto de dados para análise (ex: X_train).
        feature_names (list): Lista com os nomes das features.
        iteration (int): O número da iteração atual.
        well_name (str): O nome do poço.
        output_dir (Path): O diretório base para salvar os resultados.
    """
    print(f"\n--- [Sensitivity Analysis] Iniciando para a Iteração {iteration} - Poço {well_name} ---")

    # Usar um pequeno número de amostras para a análise ser rápida
    samples_to_analyze = X_data[:10].copy()
    if len(samples_to_analyze) == 0:
        print("[Sensitivity Analysis] Sem dados para analisar. Pulando.")
        return

    # 1. Obter as previsões de linha de base com os dados originais
    base_predictions = model.predict(samples_to_analyze, verbose=0)

    feature_importance = {}

    # 2. Iterar sobre cada feature, perturbá-la e medir o impacto
    num_features = samples_to_analyze.shape[2]
    for i in range(num_features):
        feature_name = feature_names[i]
        
        # Criar uma cópia dos dados para perturbar
        perturbed_data = samples_to_analyze.copy()
        
        # Perturbar a feature i "zerando-a" (substituindo pelo valor médio do background, ou 0 se for escalonado)
        # Para dados escalonados, 0 pode não ser a melhor referência. Vamos usar a média da feature.
        perturbation_value = np.mean(samples_to_analyze[:, :, i])
        perturbed_data[:, :, i] = perturbation_value # "Desliga" a feature
        
        # Fazer predições com a feature perturbada
        perturbed_predictions = model.predict(perturbed_data, verbose=0)
        
        # Calcular o impacto: a média da mudança absoluta no erro
        # (pode ser MSE, MAE, etc. Vamos usar a diferença absoluta na predição)
        impact = np.mean(np.abs(base_predictions - perturbed_predictions))
        feature_importance[feature_name] = impact
    
    # --- 3. Gerar e Salvar Resultados ---
    well_name_safe = well_name.replace('/', '_').replace(' ', '_')
    iter_output_dir = output_dir / f"iter_{iteration}"
    iter_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Salvando resultados em: {iter_output_dir}")
    
    # Criar um DataFrame com os resultados
    importance_df = pd.DataFrame(
        list(feature_importance.items()),
        columns=['Feature', 'Importance (Impact)']
    ).sort_values(by='Importance (Impact)', ascending=False)
    
    # Salvar o CSV
    importance_df.to_csv(iter_output_dir / "feature_importance.csv", index=False)
    
    # Gerar o gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance (Impact)'])
    plt.xlabel('Impacto na Previsão (Mudança Média Absoluta)')
    plt.title(f'Importância das Features - Poço {well_name} - Iteração {iteration}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(iter_output_dir / "feature_importance_bar.png")
    plt.close()
    
    print(f"--- [Sensitivity Analysis] Análise para a Iteração {iteration} concluída. ---")

_MODEL_CACHE: dict[str, tf.keras.Model] = {}
def _get_or_create_model(model_path: str, X_train: np.ndarray, *, architecture_name: str):
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print("Creating a NEW Model...")
        model = create_model(architecture_name=architecture_name, input_shape=X_train)
    _MODEL_CACHE[model_path] = model
    return model

def fine_tune_and_predict_well(args):  # noqa: D401
    (
        i,
        well,
        sets_i,
        X_tests_i,
        y_tests_i,
        max_train,
        scalers_i,
        model_type,
        model_path,
        fine_tune,
        cum_sum,
        control_iteration,
    ) = args

    X_train, y_train = sets_i[0], sets_i[2]
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        # print(f"Not enough data to continue training for well {well}. Skipping.")
        return None

    if model_type == "DL":
        X_train_r = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        model = _get_or_create_model(model_path, X_train_r, architecture_name="Generic")

        # Train using `train_on_batch` – no disk checkpoint during fine‑tune
        for epoch in range(25):
            model.train_on_batch(X_train_r[-2:], y_train[-2:])



        # 🔽 ANÁLISE SHAP ON-THE-FLY 🔽
        # if control_iteration in [1, 500, 1000, 1500, 2000, 2500, 3000]:
        #     # Em vez de salvar o modelo, ou além de salvar, chamamos a análise.

        #     # print('X_train_r shape', X_train_r.shape)
        #     # print('X_train_r head', X_train_r[:5])
            
        #     # --- Bloco de salvamento (pode manter se quiser os snapshots) ---
        #     # suffix = f"iter{control_iteration}_well{well.replace('/', '_')}"
        #     # save_path = Path(model_path).with_stem(f"{Path(model_path).stem}_{suffix}")
        #     # model.save(save_path)
        #     # print(f"[SAVE] Modelo salvo: {save_path}")
        #     # --- Fim do bloco de salvamento ---

        #     # --- Chamada para a Análise SHAP ---
        #     run_shap_gradient_analysis(
        #         model=model,
        #         X_data=X_train_r,
        #         feature_names=['Prod_Start_Time', 'BORE_GAS_VOL', 'ON_STREAM_HRS', 'BORE_OIL_VOL_LAG_7', 
        #                        'BORE_OIL_VOL_LAG_6', 'BORE_OIL_VOL_LAG_5', 'BORE_OIL_VOL_LAG_4', 
        #                        'BORE_OIL_VOL_LAG_3', 'BORE_OIL_VOL_LAG_2', 'BORE_OIL_VOL_LAG_1', 
        #                        'BORE_OIL_VOL_MEAN_LAG'], # Você precisará passar isso através dos 'args'
        #         iteration=control_iteration,
        #         well_name=well,
        #         output_dir=Path('SHAP') # Passar o caminho base para os resultados
        #     )
    else:
        # GBR/XGB path – still stateless
        model = train_and_evaluate_XGB(X_train, y_train, control_iteration=control_iteration, model_path = 'models/Model.json', update_rounds=10)

    # Predict
    predictions = make_predictions_for_all_wells([X_tests_i], model)[0]
    y_tests_i_list, predictions_list_i = inverse_transform_predictions(
        [y_tests_i], [predictions], [X_tests_i], model, [scalers_i], [max_train], cum_sum
    )
    return i, y_tests_i_list[0], predictions_list_i[0]


# ─── Standard library imports ────────────────────────────────────────────────────
import gc                                          # garbage collection
import logging                                     # logging utilities
from multiprocessing import Process, Pipe         # subprocess handling
from typing import Any, Dict                       # tipos para annotations

# ─── Third-party imports ────────────────────────────────────────────────────────
import numpy as np                                 # array computing
import tensorflow as tf                            # deep learning
import textwrap


def _cleanup():
    """Clear TF session and Python GC."""
    tf.keras.backend.clear_session()
    gc.collect()

def mc_dropout_predict(model, inputs, n_samples: int):
    """Mean prediction over n_samples MC‐Dropout passes."""
    preds = []
    for _ in range(n_samples):
        out = model(inputs, training=True)
        preds.append(out[0].numpy() if isinstance(out, tuple) else out.numpy())
    return np.stack(preds, 0).mean(0)



from tensorflow.keras.layers import Dense
def analyze_trend_contribution(model, verbose=True):
    """
    Analisa a contribuição das tendências e do ramo no modelo.

    Parâmetros:
        model: modelo Keras a ser analisado
        verbose: se True, emite logs de informações
    """
    dense_layer = None
    # Busca a última camada Dense com pesos
    for layer in reversed(model.layers):
        try:
            weights = layer.get_weights()
            if isinstance(layer, Dense) and len(weights) == 2:
                dense_layer = layer
                break
        except Exception:
            continue

    if dense_layer is None:
        logging.info("No suitable Dense layers found for analysis.")
        return 1, 1

    # Extrai pesos da camada Dense encontrada
    dense_weights, _ = dense_layer.get_weights()
    trend_weights = dense_weights[:1, :]
    physics_weights = dense_weights[1:, :]

    # Calcula contribuições médias
    trend_contrib = np.mean(np.abs(trend_weights))
    physics_contrib = np.mean(np.abs(physics_weights))
    total = trend_contrib + physics_contrib

    trend_pct = trend_contrib / total * 100
    physics_pct = physics_contrib / total * 100
    
    return trend_pct, physics_pct






def analyze_contributions(Qs, res, alpha, scaler_target):
    # Converte alpha para numpy:
    alpha_arr = np.asarray(alpha)    # shape: (H,) ou (batch,H)

    # Calcula Qs e res em unidades originais:
    # 1) reconstrói total_scaled = alpha*Qs + (1-alpha)*res
    total_scaled = alpha_arr * Qs + (1.0 - alpha_arr) * res

    # 2) inversão de escala conjunta, para evitar erros de transformação separada
    total_orig = scaler_target.inverse_transform(total_scaled.reshape(-1,1)).flatten()

    # 3) inversão em cada termo (se realmente quiser separar)
    Qs_orig  = scaler_target.inverse_transform((alpha_arr*Qs).reshape(-1,1)).flatten()
    res_orig = scaler_target.inverse_transform(((1-alpha_arr)*res).reshape(-1,1)).flatten()

    # Agora garanta que tudo seja não-negativo (se fizer sentido no seu domínio):
    Qs_orig = np.abs(Qs_orig)
    res_orig = np.abs(res_orig)
    denom_orig = Qs_orig + res_orig + 1e-8

    frac_i = Qs_orig / denom_orig   # array em [0,1]
    avg_frac_o = float(frac_i.mean())
    global_frac_o = float(Qs_orig.sum() / (Qs_orig.sum() + res_orig.sum()))

    logging.info(f"[Original-units] Mean point-wise physics contribution: {avg_frac_o:.2%}")
    logging.info(f"[Original-units] Global physics contribution: {global_frac_o:.2%} (alpha média: {alpha_arr.mean():.2%})")

    summary = textwrap.dedent(f"""
        === Contribution Analysis Summary (unidades originais) ===

        • Point-wise mean:       {avg_frac_o:.2%}
        • Global aggregate:      {global_frac_o:.2%} (alpha média: {alpha_arr.mean():.2%})
    """).strip()
    logging.info(summary)




# (insira logo depois dos imports internos)
def _chunk_worker(
    conn,
    build_fn,
    arch,
    kind,
    train_kwargs,
    data_inputs,
    chunk_size,
    with_snapshots,
    epochs,
    batch_size,
    patience,
    max_retries,
    skip_on_failure,
    agg_sigma="approx"  # "approx" → σ_ens² = mean(σ_i²) + var(μ_i)
):
    """Treina `chunk_size` modelos, faz ensemble (snapshots) e devolve saídas
    no formato usado historicamente pelo pipeline:

        pred_test, pred_val, q_phys, res, mu_raw, sigma, alpha, successful_models

    Campos opcionais são incluídos apenas se existirem na saída do modelo.

    Se `sigma` estiver presente o ensemble calcula automaticamente a soma de
    incerteza aleatória e epistêmica.
    """

    import numpy as np
    import tensorflow as tf
    import gc, logging

    # ------------------------------------------------------------------
    # 1.  Mapeamento tamanho → chaves
    # ------------------------------------------------------------------
    _OUT_SCHEMAS = {
        5: ("pred", "q_phys", "res", "sigma", "alpha"),
        4: ("pred", "q_phys", "res",            "alpha"),
        3: ("pred", "q_phys", "alpha"),
        1: ("pred",),
    }

    def _tuple_to_dict(out):
        if not isinstance(out, tuple):
            out = (out,)
        try:
            schema = _OUT_SCHEMAS[len(out)]
        except KeyError:
            raise ValueError(f"Saída com {len(out)} branches não suportada")
        return {k: out[i] for i, k in enumerate(schema)}

    def _online_mean(running, new, n):
        return new.copy() if running is None else running + (new - running) / n

    # ------------------------------------------------------------------
    # 2.  Ensemble via snapshots
    # ------------------------------------------------------------------
    def predict_with_snapshots(model, X):
        snaps = getattr(model, "_snapshot_weights", None)
        if not snaps:
            raise ValueError("Lista de snapshots vazia!")

        cur_weights = model.get_weights()
        outs = []
        for w in snaps:
            model.set_weights(w)
            outs.append(_tuple_to_dict(model(X, training=False)))
        model.set_weights(cur_weights)

        keys = outs[0].keys()
        stk = {k: np.stack([d[k] for d in outs], axis=0) for k in keys}

        # média de todos exceto σ
        agg = {k: stk[k].mean(axis=0) for k in keys if k != "sigma"}

        if "sigma" in keys:
            if agg_sigma == "approx":  # σ_ens² = E[σ²] + Var(μ)
                mu_stack = stk["pred"]  # usamos curva final como μ_i
                sigma_stack = stk["sigma"]
                mu_mean  = mu_stack.mean(axis=0)
                var_mu   = mu_stack.var(axis=0)
                mean_sig2 = (sigma_stack ** 2).mean(axis=0)
                agg["sigma"] = np.sqrt(mean_sig2 + var_mu)
            else:
                agg["sigma"] = stk["sigma"].mean(axis=0)
        return agg

    # ------------------------------------------------------------------
    # 3.  Dados estáticos
    # ------------------------------------------------------------------
    X_test = data_inputs["X_test"]
    X_val  = data_inputs["X_val"]

    # Agregadores online
    running_test = {}
    running_val  = {}
    running_alpha_sum = 0.0
    models_with_alpha = 0
    successful        = 0

    trend_accum = physics_accum = contrib_count = 0.0

    # ------------------------------------------------------------------
    # 4.  Loop principal
    # ------------------------------------------------------------------
    try:
        for _ in range(chunk_size):
            outs_test = outs_val = None

            # 4.1 build + treino (com retries)
            for attempt in range(max_retries + 1):
                try:
                    model, _ = build_fn(
                        arch, kind, train_kwargs, data_inputs,
                        epochs, batch_size, patience
                    )

                    if with_snapshots:
                        outs_test = predict_with_snapshots(model, X_test)
                        outs_val  = predict_with_snapshots(model, X_val)
                        try:
                            t_pct, p_pct = analyze_trend_contribution(model, verbose=False)
                            trend_accum   += t_pct
                            physics_accum += p_pct
                            contrib_count += 1
                        except Exception:
                            pass
                    else:
                        outs_test = _tuple_to_dict(model(X_test, training=False))
                        outs_val  = _tuple_to_dict(model(X_val,  training=False))
                    break  # sucesso — sai do retry
                except Exception as ex:
                    logging.warning("Attempt %d failed: %s", attempt, ex, exc_info=True)
                    tf.keras.backend.clear_session(); gc.collect()
                    if attempt == max_retries:
                        if skip_on_failure:
                            outs_test = outs_val = None
                            break
                        raise

            if outs_test is None:
                continue  # modelo pulado

            successful += 1

            # 4.2 Agregação online
            def _accumulate(dic_out, dest, suffix):
                """dest é dict running_test ou running_val"""
                for k, arr in dic_out.items():
                    # q_phys só faz sentido para test
                    if k == "q_phys" and suffix != "val":
                        continue
                    key = k if k == "q_phys" else f"{k}_{suffix}"
                    dest[key] = _online_mean(dest.get(key), arr, successful)

            _accumulate(outs_test, running_test, "test")
            _accumulate(outs_val,  running_val,  "val")

            if "alpha" in outs_test:
                running_alpha_sum += outs_test["alpha"]
                models_with_alpha += 1

            tf.keras.backend.clear_session(); gc.collect()

        # ------------------------------------------------------------------
        if successful == 0:
            raise RuntimeError("No models succeeded in this chunk")

        if with_snapshots and contrib_count > 0:
            logging.info(
                "Trend Contribution: %.2f%% | Physics Contribution: %.2f%%",
                trend_accum / contrib_count,
                physics_accum / contrib_count,
            )

        # ------------------------------------------------------------------
        # 5.  Saída no formato compatível
        # ------------------------------------------------------------------
        chunk_output = {
            "successful_models": successful,
            "alpha": (running_alpha_sum / models_with_alpha) if models_with_alpha else None,
        }
        # junta dicionários "test" e "val" mantendo nomes convencionais
        chunk_output.update(running_test)
        chunk_output.update(running_val)
        conn.send(chunk_output)

    except Exception as e:
        conn.send(e)

    finally:
        conn.close()
        tf.keras.backend.clear_session(); gc.collect()




def train_predict_chunk(
    build_fn,
    arch: str,
    kind: str,
    train_kwargs: dict,
    data_inputs: dict,
    chunk_size: int,
    with_snapshots: int,
    epochs: int,
    batch_size: int,
    patience: int,
    max_retries: int,
    skip_on_failure: bool
) -> dict:
    logging.info(f"→ Launching chunk of {chunk_size} models")
    parent_conn, child_conn = Pipe()
    p = Process(
        target=_chunk_worker,
        args=(
            child_conn,
            build_fn,
            arch,
            kind,
            train_kwargs,
            data_inputs,
            chunk_size,
            with_snapshots,
            epochs,
            batch_size,
            patience,
            max_retries,
            skip_on_failure
        )
    )
    p.start()
    result = parent_conn.recv()
    p.join()
    if isinstance(result, Exception):
        logging.error("Chunk worker raised exception", exc_info=True)
        raise result
    logging.info(f"← Chunk complete, {result['successful_models']} models aggregated")
    return result




