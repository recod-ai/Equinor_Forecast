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


# src/common/config_wells.py

from pathlib import Path
from data.data_preparation import apply_custom_kalman_filter # Supondo que esta função exista

# --- Constantes do Módulo ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

# src/common/config_wells.py

from pathlib import Path
from data.data_preparation import apply_custom_kalman_filter  # Assuming this function exists

# --- Module Constants ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

# The variable mapping is a useful constant to keep here.
UNISIM_VARIABLE_MAPPING = {
    'Well pressure': 'PWFO',
    'Oil flow': 'QOOB',
    'Water flow': 'QWOB',
    'Liquid flow (oil + water)': 'QLOB',
    'Gas flow': 'QGOB',
}

# --- Main Function (Source of Truth) ---

def get_data_sources(opsd_type: str = "wind") -> list[dict]:
    """
    Generates the list of data source configurations.
    This function is now the single source of truth for these configurations.

    Args:
        opsd_type (str): The OPSD data type ('wind', 'solar', 'load').
                         Default is 'wind'.

    Returns:
        list[dict]: A list of dictionaries, each describing a data source.
    """
    # OPSD logic is encapsulated within the function
    if opsd_type == "load":
        opsd_serie_name = "GB_GBN_load_actual_entsoe_transparency"
    else:
        opsd_serie_name = f"GB_GBN_{opsd_type}_generation_actual"

    # The data sources list definition now lives inside here.
    return [
        # --- VOLVE ---
        {
            "name":  "VOLVE",
            # FIX: The "wells" key was duplicated. I combined the lists.
            # If the intention was to use only one, change it here.
            "wells": ["15/9-F-14", "15/9-F-12", "15/9-F-11", "15/9-F-15 D"],
            # "wells": ["15/9-F-14"],
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
            "features": ["QOOB", "QGOB", "QLOB", "PWFO", "QWOB", "Tempo_Inicio_Prod"],
            "filter_postprocess": apply_custom_kalman_filter,
        },
        # --- OPSD ---
        {
            "name":  "OPSD",
            "wells": [opsd_type],
            "load_params": {
                "data_path":  DATA_DIR / "OPSD" / "time_series_30min_singleindex.csv",
                "serie_name": opsd_serie_name,
                "remove_zeros": True,
            },
            "model_path": BASE_DIR / "OPSD_MODELS" / f"best_model_OPSD_{opsd_type}.keras",
            "target_column": f"GB_GBN_{opsd_type}_generation_tax",
            "variable_mapping": None,
            "features": [
                "Tempo_Inicio_Prod",
                f"GB_GBN_{opsd_type}_generation_tax",
                opsd_serie_name,
            ],
            "filter_postprocess": apply_custom_kalman_filter,
        },
    ]

# --- Compatibility Bridge (Intermediate Step) ---
# To ensure that other parts of the code that use `from ... import DATA_SOURCES`
# continue to work, we create the variable by calling our new function.
# The default behavior of the old code was to use 'wind', so we keep that.
DATA_SOURCES = get_data_sources(opsd_type="wind")

