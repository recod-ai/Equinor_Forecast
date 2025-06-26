#!/usr/bin/env python3
"""
reorganize_project.py

This script reorganizes the current project into the following structure:

project_root/
├── data/                        # Raw datasets (unchanged)
│
├── notebooks/                   # Notebooks for experiments
│   ├── darts/                   # Notebooks related to the Darts regression approach
│   ├── forecast/                # Notebooks for forecasting experiments (XGB, DL, etc.)
│   └── legacy/                  # Legacy notebooks and scripts
│
├── src/                         # Production / reusable code
│   ├── __init__.py              # (Optional) Package initializer
│   ├── data/                    # Data-related code (data_loading.py, data_preparation.py)
│   ├── models/                  # Model definitions and registries
│   │   ├── my_models/           # In-house regression approaches
│   │   └── darts_models/        # Darts-specific models (if needed)
│   ├── training/                # Training routines and utilities (models_forecast.py, train_utils.py, darts_parallel.py)
│   ├── evaluation/              # Evaluation metrics and plotting routines (evaluation.py)
│   ├── prediction/              # Prediction/inference utilities (prediction_utils.py)
│   └── utils/                   # Miscellaneous utilities (utilities.py)
│
├── outputs/                     # Artifacts from training runs
│   ├── checkpoints/
│   ├── darts_logs/
│   ├── OPSD_MODELS/
│   ├── UNISIM_MODELS/
│   └── VOLVE_MODELS/
│
├── requirements.txt             # Dependencies list
├── README.md                    # Project overview
└── setup.py                     # Packaging script (if applicable)
"""

import os
import shutil

# Define absolute paths relative to the project root.
# If running from the project root, os.getcwd() gives the current directory.
PROJECT_ROOT = os.getcwd()

# New folder structure (relative to project root)
new_dirs = [
    os.path.join(PROJECT_ROOT, 'data'),
    os.path.join(PROJECT_ROOT, 'notebooks', 'darts'),
    os.path.join(PROJECT_ROOT, 'notebooks', 'forecast'),
    os.path.join(PROJECT_ROOT, 'notebooks', 'legacy'),
    os.path.join(PROJECT_ROOT, 'src', 'data'),
    os.path.join(PROJECT_ROOT, 'src', 'models', 'my_models'),
    os.path.join(PROJECT_ROOT, 'src', 'models', 'darts_models'),
    os.path.join(PROJECT_ROOT, 'src', 'training'),
    os.path.join(PROJECT_ROOT, 'src', 'evaluation'),
    os.path.join(PROJECT_ROOT, 'src', 'prediction'),
    os.path.join(PROJECT_ROOT, 'src', 'utils'),
    os.path.join(PROJECT_ROOT, 'outputs', 'checkpoints'),
    os.path.join(PROJECT_ROOT, 'outputs', 'darts_logs'),
    os.path.join(PROJECT_ROOT, 'outputs', 'OPSD_MODELS'),
    os.path.join(PROJECT_ROOT, 'outputs', 'UNISIM_MODELS'),
    os.path.join(PROJECT_ROOT, 'outputs', 'VOLVE_MODELS'),
]

print("Creating new directories...")
for d in new_dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created or exists: {d}")

# Mapping of source items (files or folders) to their new destination folder.
# Be sure to adjust these names if needed.
# Format: "old_relative_path": "new_relative_directory"
moves = {
    # --- Notebooks ---
    "DARTS.ipynb": os.path.join("notebooks", "darts"),
    "DARTS_Hybrid.ipynb": os.path.join("notebooks", "darts"),
    "forecast_DL.ipynb": os.path.join("notebooks", "forecast"),
    "forecast_XGB.ipynb": os.path.join("notebooks", "forecast"),
    "Oil_Forecast_DL.ipynb": os.path.join("notebooks", "forecast"),
    "Oil_Forecast_XGBoost.ipynb": os.path.join("notebooks", "forecast"),
    
    # legacy notebooks and scripts
    "Custom_Darts.ipynb": os.path.join("notebooks", "legacy"),
    "forecast_OPSD.ipynb": os.path.join("notebooks", "legacy"),
    "IPR solutions.ipynb": os.path.join("notebooks", "legacy"),
    "Legacy_PIN_ARPS.ipynb": os.path.join("notebooks", "legacy"),
    "organize_data_legacy.py": os.path.join("notebooks", "legacy"),
    "utils_legacy.py": os.path.join("notebooks", "legacy"),

    # --- Source code files to be moved into src/ ---
    "data_loading.py": os.path.join("src", "data"),
    "data_preparation.py": os.path.join("src", "data"),
    "evaluation.py": os.path.join("src", "evaluation"),
    "models_forecast.py": os.path.join("src", "training"),
    "train_utils.py": os.path.join("src", "training"),
    "darts_parallel.py": os.path.join("src", "training"),
    "prediction_utils.py": os.path.join("src", "prediction"),
    "utilities.py": os.path.join("src", "utils"),
    "__init__.py": os.path.join("src"),
    
    # --- Folders to be moved ---
    "models": os.path.join("src", "models", "my_models"),  # Your in-house models are currently in models/
    # If you have Darts-specific models in the current "models" folder, you may need to split them.
    
    # --- Outputs directories ---
    "checkpoints": os.path.join("outputs", "checkpoints"),
    "darts_logs": os.path.join("outputs", "darts_logs"),
    "OPSD_MODELS": os.path.join("outputs", "OPSD_MODELS"),
    "UNISIM_MODELS": os.path.join("outputs", "UNISIM_MODELS"),
    "VOLVE_MODELS": os.path.join("outputs", "VOLVE_MODELS"),
}

def move_item(src_rel, dest_rel):
    src_path = os.path.join(PROJECT_ROOT, src_rel)
    dest_dir = os.path.join(PROJECT_ROOT, dest_rel)
    
    if not os.path.exists(src_path):
        print(f"WARNING: {src_path} does not exist.")
        return

    # Determine final destination path
    # If src is a file, preserve its filename; if a folder, move the entire folder.
    item_name = os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, item_name)
    
    print(f"Moving {src_path} -> {dest_path}")
    try:
        shutil.move(src_path, dest_path)
    except Exception as e:
        print(f"Error moving {src_path} to {dest_path}: {e}")

# Process all moves defined above
print("\nMoving files and folders...")
for src_item, dest_folder in moves.items():
    move_item(src_item, dest_folder)

print("\nProject reorganization complete!")
