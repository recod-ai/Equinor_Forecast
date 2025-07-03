# src/common/config_wells.py
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
from data.data_preparation import apply_custom_kalman_filter

# Define a mapping for UNISIM-II-H variables
UNISIM_VARIABLE_MAPPING = {
    'Well pressure': 'PWFO',
    'Oil flow': 'QOOB',
    'Water flow': 'QWOB',
    'Liquid flow (oil + water)': 'QLOB',
    'Gas flow': 'QGOB',
}


def get_data_sources(opsd_type=None):
    if opsd_type is None:
        opsd_type = "wind"  # padrão

    # Casos especiais para a chave de série do OPSD
    if opsd_type == "load":
        serie_name = "GB_GBN_load_actual_entsoe_transparency"
    else:
        serie_name = f"GB_GBN_{opsd_type}_generation_actual"

    return [
        # --- VOLVE ---
        {
            "name":  "VOLVE",
            "wells": ["15/9-F-14", "15/9-F-12", "15/9-F-11", "15/9-F-15 D"],
            "load_params": {
                "data_path":  DATA_DIR / "volve" / "Volve_Equinor.csv",
                "serie_name": "BORE_OIL_VOL",
                "add_physical_features": False,
            },
            "model_path": BASE_DIR / "VOLVE_MODELS" / "best_disruptive_model_VOLVE.keras",
            "target_column": "BORE_OIL_VOL",
            "variable_mapping": None,
            # "features": [
            #     "BORE_GAS_VOL", "CE", "delta_P", "PI", "AVG_DOWNHOLE_PRESSURE",
            #     "BORE_WAT_VOL", "ON_STREAM_HRS", "Tempo_Inicio_Prod",
            #     "Taxa_Declinio", "BORE_OIL_VOL",
            # ],
            "features": [
                "Tempo_Inicio_Prod", "BORE_GAS_VOL", "ON_STREAM_HRS", "BORE_OIL_VOL"
            ],
            "filter_postprocess": apply_custom_kalman_filter,
        },
        # --- UNISIM_IV ---
        {
            "name": "UNISIM_IV",
            "wells": ["P11"],
            "load_params": {
                "data_path": DATA_DIR / "UNISIM-IV-2026" / "Well_{well}_UNISIM-IV.csv",
                "serie_name": "BORE_OIL_VOL",
            },
            "model_path": BASE_DIR / "UNISIM-IV_MODELS" / "best_disruptive_model_UNISIM-IV.keras",
            "target_column": "BORE_OIL_VOL",
            "variable_mapping": None,
            "features": [
                "PI", "CE", "BORE_GAS_VOL", "AVG_DOWNHOLE_PRESSURE",
                "AVG_WHP_P", "Tempo_Inicio_Prod", "Taxa_Declinio", "BORE_OIL_VOL",
            ],
            "filter_postprocess": apply_custom_kalman_filter,
        },
        # --- UNISIM ---
        {
            "name":  "UNISIM",
            "wells": [
                "Prod-1", "Prod-2", "Prod-3", "Prod-4",
                "Prod-5", "Prod-6", "Prod-7", "Prod-8", "Prod-9", "Prod-10",
            ],
            "load_params": {
                "data_path":  DATA_DIR / "unisim" / "production.csv",
                "serie_name": "QOOB",
                "remove_zeros": True,
            },
            "model_path": BASE_DIR / "UNISIM_MODELS" / "best_disruptive_model_UNISIM.keras",
            "target_column": "QOOB",
            "variable_mapping": None,
            "features": ["QOOB", "Tempo_Inicio_Prod"],
            "filter_postprocess": apply_custom_kalman_filter,
        },
        # --- OPSD ---
        {
            "name":  "OPSD",
            "wells": [opsd_type],
            "load_params": {
                "data_path":  DATA_DIR / "OPSD" / "time_series_30min_singleindex.csv",
                "serie_name": serie_name,
                "remove_zeros": True,
            },
            "model_path": BASE_DIR / "OPSD_MODELS" / f"best_model_OPSD_{opsd_type}.keras",
            "target_column": f"GB_GBN_{opsd_type}_generation_tax",
            "variable_mapping": None,
            "features": [
                "Tempo_Inicio_Prod",
                f"GB_GBN_{opsd_type}_generation_tax",
                serie_name,
            ],
            "filter_postprocess": apply_custom_kalman_filter,
        },
    ]



# Static for other components
opsd_filed = 'wind'  # wind, solar or load

# --- Ajuste da série para o caso especial "load" ---
if opsd_filed == "load":
    serie_name = "GB_GBN_load_actual_entsoe_transparency"
else:
    serie_name = f"GB_GBN_{opsd_filed}_generation_actual"

    
DATA_SOURCES = [
    {
        "name":  "VOLVE",
        "wells": ["15/9-F-14", "15/9-F-12", "15/9-F-11", "15/9-F-15 D"],
        "wells": ["15/9-F-14"],
        "load_params": {
            "data_path":  DATA_DIR / "volve" / "Volve_Equinor.csv",
            "serie_name": "BORE_OIL_VOL",
            "add_physical_features": False,
        },
        "model_path": BASE_DIR / "VOLVE_MODELS" / "best_disruptive_model_VOLVE.keras",
        "target_column": "BORE_OIL_VOL",
        "variable_mapping": None,
        "features": [
            "BORE_GAS_VOL", "CE", "delta_P", "PI", "AVG_DOWNHOLE_PRESSURE",
            "BORE_WAT_VOL", "ON_STREAM_HRS", "Tempo_Inicio_Prod",
            "Taxa_Declinio", "BORE_OIL_VOL",
        ],
        "filter_postprocess": apply_custom_kalman_filter,
    },
    # "UNISIM-IV"
    {
        "name": "UNISIM_IV",
        "wells": ["P11"],
        "load_params": {
            # apontar para a pasta onde estão seus CSVs: well_P11_UNISIM-IV.csv, etc.
            "data_path": DATA_DIR / "UNISIM-IV-2026" / "Well_{well}_UNISIM-IV.csv",
            "serie_name": "BORE_OIL_VOL",
        },
        "model_path": BASE_DIR / "UNISIM-IV_MODELS" / "best_disruptive_model_UNISIM-IV.keras",
        "target_column": "BORE_OIL_VOL",
        "variable_mapping": None,
        "features": [
                "PI",
                "CE",
                "BORE_GAS_VOL",
                "AVG_DOWNHOLE_PRESSURE",
                "AVG_WHP_P",
                "Tempo_Inicio_Prod",
                "Taxa_Declinio",
                "BORE_OIL_VOL",
        ],
        "filter_postprocess": apply_custom_kalman_filter,
    },
    
    # --- UNISIM ---
    {
        "name":  "UNISIM",
        "wells": [  # não duplique a chave!
            "Prod-1", "Prod-2", "Prod-3", "Prod-4",
            "Prod-5", "Prod-6", "Prod-7", "Prod-8", "Prod-9", "Prod-10",
        ],
        "load_params": {
            "data_path":  DATA_DIR / "unisim" / "production.csv",
            "serie_name": "QOOB",
            "remove_zeros": True,
        },
        "model_path": BASE_DIR / "UNISIM_MODELS" / "best_disruptive_model_UNISIM.keras",
        "target_column": "QOOB",
        "variable_mapping": None,
        "features": ["QOOB", "QGOB", "QLOB", "PWFO", "QWOB", "Tempo_Inicio_Prod"],
        "filter_postprocess": apply_custom_kalman_filter,
    },
    # --- OPSD ---
    {
        "name":  "OPSD",
        "wells": [opsd_filed],
        "load_params": {
            "data_path":  DATA_DIR / "OPSD" / "time_series_30min_singleindex.csv",
            "serie_name": serie_name,
            "remove_zeros": True,
        },
        "model_path": BASE_DIR / "OPSD_MODELS" / f"best_model_OPSD_{opsd_filed}.keras",
        "target_column": f"GB_GBN_{opsd_filed}_generation_tax",
        "variable_mapping": None,
        "features": [
            "Tempo_Inicio_Prod",
            f"GB_GBN_{opsd_filed}_generation_tax",
            serie_name,
        ],
        "filter_postprocess": apply_custom_kalman_filter,
    },
]
