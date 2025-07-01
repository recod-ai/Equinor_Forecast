import argparse
from pathlib import Path

# Importa a lógica principal do pipeline.py
from pipeline import WellForecastPipeline

# Importa as configurações (onde DATA_SOURCES está definido)
from common.config_wells import DATA_SOURCES

def main():
    """Ponto de entrada principal para executar o pipeline para um único poço."""
    parser = argparse.ArgumentParser(description="Run well forecast pipeline for a specific well.")
    parser.add_argument("--well", type=str, required=True, help="Name of the well to process.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset.")
    args = parser.parse_args()

    try:
        data_source = next(item for item in DATA_SOURCES if item["name"] == args.dataset)
    except StopIteration:
        print(f"ERROR: Dataset '{args.dataset}' not found in configuration.")
        return

    print(f"--- Starting Pipeline for Well: {args.well} in Dataset: {args.dataset} ---")
    
    # Cria um caminho de modelo específico para este poço para evitar conflitos
    base_model_path = Path(data_source["model_path"])
    well_specific_model_path = base_model_path.with_stem(f"{base_model_path.stem}_{args.well.replace('/', '_')}")
    
    # Instancia e executa o pipeline com os parâmetros corretos
    pipeline_runner = WellForecastPipeline(
        dataset=data_source['name'],
        wells=[args.well],  # Passa uma lista com o único poço a ser processado
        model_path=well_specific_model_path,
        serie_name=data_source['load_params']["serie_name"],
        data_path=data_source['load_params']['data_path'],
        forecast_steps=56,
        window_size=7,
        model_type="DL",
        sample_time=1,
        train_windows=150,
        fine_tuning_windows=150,
        cum_sum=True,
        data_loader_kwargs=data_source.get("load_params"),
        variable_mapping=data_source.get("variable_mapping"),
        filter_postprocess=data_source.get("filter_postprocess"),
        architecture_name='Generic'
    )
    pipeline_runner.run()
    
    print(f"--- Finished Pipeline for Well: {args.well} ---")

if __name__ == "__main__":
    main()