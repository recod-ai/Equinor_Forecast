import os
import shutil

from pathlib import Path

import glob
import logging


# =============================================================================
# Helper Functions (General)
# =============================================================================

def clean_checkpoint_files(checkpoint_dir="model_checkpoint"):
    """Remove all files in the checkpoint directory."""
    for file in glob.glob(os.path.join(checkpoint_dir, "*")):
        os.remove(file)
    logging.info("Checkpoint directory cleaned.")
    



def clean_checkpoint_folder(checkpoint_folder='model_checkpoint', max_models=100) -> None:
    """
    Cleans the checkpoint folder by removing all its contents if it contains more than max_models items.

    Parameters:
        checkpoint_folder (str): Path to the folder containing model checkpoints.
        max_models (int): Maximum allowed number of model checkpoints before cleaning.
    """
    folder = Path(checkpoint_folder)
    if not folder.exists():
        return  # Nothing to clean if the folder doesn't exist

    items = list(folder.iterdir())
    if len(items) > max_models:
        for item in items:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

# Example usage: at the beginning of your program
if __name__ == '__main__':
    checkpoint_path = 'model_checkpoint'
    max_allowed_models = 10  # Set your threshold here
    clean_checkpoint_folder(checkpoint_path, max_allowed_models)



def generate_jobs(data_sources, wells, experiments):
    """Generate a list of experiment jobs with a unique counter."""
    jobs = []
    exp_counter = 1
    for config in data_sources:
        for well in wells:
            for exp_params in experiments:
                jobs.append((config, well, exp_params, exp_counter))
                exp_counter += 1
    return jobs


def delete_all_files_in_folder(folder_path):
    """
    Apaga todos os arquivos e subdiretórios dentro da pasta especificada.
    
    Args:
        folder_path (str): Caminho para a pasta onde os arquivos devem ser apagados.
    
    Returns:
        None
    """
    # Verifica se o diretório existe e é uma pasta
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Itera sobre todos os arquivos e subdiretórios na pasta
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Verifica se é um arquivo ou um link simbólico e apaga
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                    print(f"Arquivo {file_path} apagado com sucesso.")
                # Se for um diretório, apaga recursivamente
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Diretório {file_path} apagado com sucesso.")
            except Exception as e:
                print(f"Erro ao apagar {file_path}. Motivo: {e}")
    else:
        print(f"Pasta {folder_path} não encontrada ou não é um diretório.")

        
from rich.console import Console
from rich.text import Text
import pyfiglet
        
def print_style(text):
    # Cria um console rich para exibir o texto estilizado
    console = Console()

    # Gera o texto em ASCII art com pyfiglet
    ascii_art = pyfiglet.figlet_format(text, font="standard")

    # Cria um objeto Text para adicionar cor e estilo
    styled_text = Text(ascii_art, style="italic blue")


    # Imprime o texto estilizado
    console.print(styled_text)
    
    
    
from typing import Any, Callable, Dict, List

def apply_filter_to_predictions(
    y_pred_list: List[List[float]],
    filter_function: Callable
) -> List[List[float]]:
    """
    Applies a filter function to the prediction lists for each well.

    Parameters:
    - y_pred_list (List[List[float]]): List of prediction lists per well.
    - filter_function (Callable): The filtering function to apply.

    Returns:
    - List[List[float]]: The filtered prediction lists.
    """
    y_pred_list_filter = []
    for current_data in y_pred_list:
        if current_data:
            filtered_data = filter_function(current_data)
            y_pred_list_filter.append(filtered_data.tolist())
        else:
            y_pred_list_filter.append([])
    print("Applied Filter")
    return y_pred_list_filter



import numpy as np
import pandas as pd

def check_data(*datasets):
    all_ok = True
    
    for i, data in enumerate(datasets):
        name = f'dataset_{i+1}'
        issues_found = False
        
        print(f'Checking {name}...')
        
        if isinstance(data, np.ndarray):
            if np.isnan(data).any():
                nan_positions = np.argwhere(np.isnan(data))
                print(f'  ❌ NaN detected in {name}, count: {nan_positions.shape[0]}')
                print(f'  NaN positions (first 10 shown):\n {nan_positions[:10]}')
                issues_found = True
            if np.isinf(data).any():
                inf_positions = np.argwhere(np.isinf(data))
                print(f'  ❌ Infinite values detected in {name}, count: {inf_positions.shape[0]}')
                print(f'  Infinite positions (first 10 shown):\n {inf_positions[:10]}')
                issues_found = True
            if np.abs(data).max() > 1e6:
                print(f'  ⚠️ Potential outliers detected in {name}')
                issues_found = True
        
        elif isinstance(data, pd.DataFrame):
            nan_count = data.isna().sum().sum()
            if nan_count > 0:
                nan_positions = data.isna()
                print(f'  ❌ NaN detected in {name}, count: {nan_count}')
                print(f'  NaN positions:\n{nan_positions[nan_positions].stack().index.tolist()[:10]}')
                issues_found = True
            if np.isinf(data.to_numpy()).sum() > 0:
                inf_positions = np.isinf(data.to_numpy())
                print(f'  ❌ Infinite values detected in {name}')
                print(f'  Infinite positions:\n{list(zip(*np.where(inf_positions)))[:10]}')
                issues_found = True
            if ((data.dtypes != 'object') & (data.abs() > 1e6).any()).any():
                print(f'  ⚠️ Potential outliers detected in {name}')
                issues_found = True
        
        if not issues_found:
            print(f'  ✅ No issues detected in {name}')
        else:
            all_ok = False
        
    if all_ok:
        print("🎉 All datasets passed the check with no issues!")
    else:
        print("⚠️ Some issues were found in the datasets.")

import os
import joblib
import tensorflow as tf


# =====================================================
# Utility Functions for Scaling Inversion
# =====================================================
def load_scaler(filepath: str):
    """
    Loads a scaler from a pickle file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scaler file not found at {filepath}.")
    scaler = joblib.load(filepath)
    tf.print(f"Scaler successfully loaded from {filepath}.")
    return scaler

def invert_feature_scaling(feature_scaled, scaler_X_mean, scaler_X_std, feature_idx: int):
    """
    Inverts the scaling of a feature given its index in the scaler arrays.
    
    Parameters:
      - feature_scaled: Scaled feature tensor.
      - scaler_X_mean: Mean values from the input scaler.
      - scaler_X_std: Standard deviation values from the input scaler.
      - feature_idx: Index of the feature in the scaler arrays.
      
    Returns:
      - The original (unscaled) feature.
    """
    feature_mean = scaler_X_mean[feature_idx]
    feature_std = scaler_X_std[feature_idx]
    return feature_scaled * feature_std + feature_mean

def get_center_and_scale(scaler, as_tf=True, dtype=tf.float32):
    """
    Devolve (center, scale) para qualquer scaler compatível com scikit-learn.
    Se as_tf=True, retorna tf.Tensor; caso contrário, retorna NumPy ndarray.
    center = média (Standard) ou mediana (Robust) ou quantil mediano (QuantileTransformer)  
    scale  = desvio-padrão (Standard) ou IQR (Robust) ou escala do QuantileTransformer  
    """
    # 1) Extrair arrays NumPy
    if hasattr(scaler, "mean_"):          # StandardScaler
        center_np = scaler.mean_
    elif hasattr(scaler, "center_"):      # RobustScaler
        center_np = scaler.center_
    elif hasattr(scaler, "quantiles_"):   # QuantileTransformer
        # para QuantileTransformer, o "quantiles_" é um array onde o elemento do meio é a mediana
        n = len(scaler.quantiles_) // 2
        center_np = scaler.quantiles_[n]
    else:
        raise AttributeError("Scaler sem atributo de centro conhecido")

    if not hasattr(scaler, "scale_"):
        raise AttributeError("Scaler sem atributo scale_")
    scale_np = scaler.scale_

    if as_tf:
        center_tf = tf.constant(center_np, dtype=dtype)
        scale_tf  = tf.constant(scale_np,  dtype=dtype)
        return center_tf, scale_tf
    else:
        return center_np, scale_np