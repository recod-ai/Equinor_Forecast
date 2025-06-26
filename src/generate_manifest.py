# Arquivo: src/generate_manifest.py

import pandas as pd
import uuid
from itertools import product
from pathlib import Path
import argparse
from typing import List

# Importa os carregadores de configuração e a lista de datasets
from config_loader import load_experiment_configs
from common.config_wells import DATA_SOURCES

# ==============================================================================
# CONTROLE DE GERAÇÃO DO MANIFESTO
# ==============================================================================

# Para incluir APENAS datasets específicos, liste-os aqui.
# Se a lista estiver vazia, todos os datasets serão considerados (exceto os excluídos).
# Exemplo: INCLUDE_DATASETS = ["UNISIM", "VOLVE"]
INCLUDE_DATASETS: List[str] = []

# Para EXCLUIR datasets específicos, liste-os aqui.
EXCLUDE_DATASETS: List[str] = ["UNISIM-IV"]

# ==============================================================================
# LÓGICA DE GERAÇÃO
# ==============================================================================

def generate_job_matrix():
    """
    Gera a matriz completa de jobs a partir das configurações, aplicando
    filtros de inclusão e exclusão de datasets.
    """
    print("INFO: Carregando todas as configurações de experimento...")
    
    try:
        exp_configs = load_experiment_configs(config_dir="experiment_configs")
        architectures = exp_configs["architectures"]
        hyperparams = exp_configs["hyperparameters"]
    except FileNotFoundError as e:
        print(f"ERRO: Não foi possível carregar os arquivos de configuração. {e}")
        return pd.DataFrame()

    # ★ NOVO: Lógica de filtragem de datasets
    filtered_data_sources = DATA_SOURCES
    
    if INCLUDE_DATASETS:
        filtered_data_sources = [
            ds for ds in filtered_data_sources if ds["name"] in INCLUDE_DATASETS
        ]
        print(f"INFO: Filtrando para incluir apenas os datasets: {INCLUDE_DATASETS}")

    if EXCLUDE_DATASETS:
        filtered_data_sources = [
            ds for ds in filtered_data_sources if ds["name"] not in EXCLUDE_DATASETS
        ]
        print(f"INFO: Excluindo os datasets: {EXCLUDE_DATASETS}")

    print("INFO: Construindo a matriz de jobs...")
    job_list = []

    # O loop agora itera sobre a lista de datasets já filtrada
    for arch_config, hp_config in product(architectures, hyperparams):
        for dataset_config in filtered_data_sources:
            dataset_name = dataset_config["name"]
            
            if dataset_name == "OPSD":
                opsd_types = ['solar', 'wind', 'load']
                for opsd_type in opsd_types:
                    job_list.append({
                        "architecture_id": arch_config.architecture_id,
                        "hyperparam_id": hp_config.hyperparam_id,
                        "dataset": dataset_name,
                        "well": opsd_type
                    })
            else:
                for well_name in dataset_config["wells"]:
                    job_list.append({
                        "architecture_id": arch_config.architecture_id,
                        "hyperparam_id": hp_config.hyperparam_id,
                        "dataset": dataset_name,
                        "well": well_name
                    })

    if not job_list:
        print("AVISO: Nenhum job foi gerado. Verifique os arquivos de configuração e os filtros.")
        return pd.DataFrame()

    # --- Criação do DataFrame (sem mudanças a partir daqui) ---
    manifest_df = pd.DataFrame(job_list)
    manifest_df["job_id"] = [str(uuid.uuid4()) for _ in range(len(manifest_df))]
    manifest_df["window_size"] = 7
    manifest_df["status"] = "pending"
    # ... (resto das colunas de metadados) ...
    
    column_order = [
        "job_id", "architecture_id", "hyperparam_id", "dataset", "well", "window_size",
        "status", "created_at", "started_at", "finished_at", "smape", "mae", "notes",
        "output_notebook_path", "log_path", "results_path"
    ]
    # Adiciona colunas que podem não existir ainda para garantir a ordem
    for col in column_order:
        if col not in manifest_df.columns:
            manifest_df[col] = None
            
    manifest_df = manifest_df[column_order]

    print(f"INFO: Matriz gerada com sucesso. Total de jobs: {len(manifest_df)}")
    return manifest_df

def main():
    """
    Ponto de entrada do script para gerar e salvar o manifesto de experimentos.
    """
    parser = argparse.ArgumentParser(
        description="Gera um manifesto de jobs para a execução de experimentos em lote.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output/manifest.csv",
        help="Caminho do arquivo CSV para salvar o manifesto gerado."
    )
    args = parser.parse_args()

    # Garante que o diretório de saída exista
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Gera a matriz de jobs
    manifest_df = generate_job_matrix()

    if not manifest_df.empty:
        # Salva o manifesto em um arquivo CSV
        manifest_df.to_csv(output_path, index=False)
        print(f"✅ Manifesto de jobs salvo com sucesso em: {output_path}")

if __name__ == "__main__":
    main()