#src/training/train_utils.py
# --------------------------------------------------------------------------------------
# 1. FUTURE IMPORTS
# --------------------------------------------------------------------------------------
# Ensures that type hints are processed lazily, which is a best practice.
from __future__ import annotations

# --------------------------------------------------------------------------------------
# 2. STANDARD LIBRARY IMPORTS
# --------------------------------------------------------------------------------------
# Grouped and alphabetized for clarity.
import gc
import json
import logging
import math
import os
import random
import textwrap
import time
import warnings
from functools import partial
from multiprocessing import Pipe, Process
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import traceback

# --------------------------------------------------------------------------------------
# 3. THIRD-PARTY IMPORTS
# --------------------------------------------------------------------------------------
# All external dependencies, grouped and alphabetized.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances, silhouette_score

# --------------------------------------------------------------------------------------
# 4. LOCAL APPLICATION IMPORTS
# --------------------------------------------------------------------------------------
# Your own project's modules, grouped by top-lefvel package.
from forecast_pipeline.config import (
    ENABLE_DETERMINISM,
    CANON_FEATURES,
    INITIAL_PRESSURE,
    SHAP_ANALYSIS,
    VARIABLE_MAPPING,
    _UNISIM_IV_MAP,
    bar2psi,
    kpa2psi,
    m3d2scfd,
    m3d2stbd,
    psi2kpa,
)
from forecast_pipeline.logging_utils import get_logger, log_context, phase
from prediction.prediction_utils import (
    inverse_transform_predictions,
    make_predictions_for_all_wells,
)
from training.models_forecast import (
    train_and_evaluate_disruptive,
    train_and_evaluate_XGB,
)
from training.train_darts import main_train_darts_model


# --------------------------------------------------------------------------------------
# 5. EXECUTABLE STATEMENTS (handle with care)
# --------------------------------------------------------------------------------------
# This is an action, not an import. Running it at the top level of a module
# can have side effects. It's often better to call this inside a specific function
# right before you build a new model, to ensure a clean state for that task.
# tf.keras.backend.clear_session()
# logging.info("TensorFlow Keras session cleared.") # Example of logging the action


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

    # O GradientExplainer retorna um SHAP value por neurônio de saída.
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
        if control_iteration in [1, 500, 1000, 1500, 2000, 2500, 3000] and SHAP_ANALYSIS:
            
            # --- Bloco de salvamento (pode manter se quiser os snapshots) ---
            # suffix = f"iter{control_iteration}_well{well.replace('/', '_')}"
            # save_path = Path(model_path).with_stem(f"{Path(model_path).stem}_{suffix}")
            # model.save(save_path)
            # print(f"[SAVE] Modelo salvo: {save_path}")

            # --- Chamada para a Análise SHAP ---
            run_shap_gradient_analysis(
                model=model,
                X_data=X_train_r,
                feature_names=['Prod_Start_Time', 'BORE_GAS_VOL', 'ON_STREAM_HRS', 'BORE_OIL_VOL_LAG_7', 
                               'BORE_OIL_VOL_LAG_6', 'BORE_OIL_VOL_LAG_5', 'BORE_OIL_VOL_LAG_4', 
                               'BORE_OIL_VOL_LAG_3', 'BORE_OIL_VOL_LAG_2', 'BORE_OIL_VOL_LAG_1', 
                               'BORE_OIL_VOL_MEAN_LAG'], # Você precisará passar isso através dos 'args'
                iteration=control_iteration,
                well_name=well,
                output_dir=Path('SHAP') # Passar o caminho base para os resultados
            )
    else:
        # GBR/XGB path – still stateless
        model = train_and_evaluate_XGB(X_train, y_train, control_iteration=control_iteration, model_path = 'models/Model.json', update_rounds=10)

    # Predict
    predictions = make_predictions_for_all_wells([X_tests_i], model)[0]
    y_tests_i_list, predictions_list_i = inverse_transform_predictions(
        [y_tests_i], [predictions], [X_tests_i], model, [scalers_i], [max_train], cum_sum
    )
    return i, y_tests_i_list[0], predictions_list_i[0]


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



def reseed_everything():
    seed = int(time.time() * 1e6) % (2**31 - 1)
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



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
    learning_rate: float,
    max_retries: int,
    skip_on_failure: bool
) -> dict:
    """
    Spawns a worker process to train a chunk of models and aggregate predictions.
    Logging is concise: start/end + key decisions; no metric logs.
    """
    logger = get_logger(__name__)
    if arch.startswith("Darts_"):
        build_fn = main_train_darts_model

    with log_context(arch=arch, kind=kind, chunk_size=chunk_size, epochs=epochs, batch_size=batch_size, lr=learning_rate):
        with phase(logger, "train_predict_chunk"):
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
                    learning_rate,
                    max_retries,
                    skip_on_failure,
                )
            )
            logger.info("decision=spawn_worker impl=%s", "darts" if arch.startswith("Darts_") else "tf")
            p.start()
            result = parent_conn.recv()
            p.join()

            if isinstance(result, Exception):
                logger.error("worker_failed", exc_info=True)
                raise result

            logger.info("chunk_done successful_models=%s", result.get("successful_models"))
            return result



# ==============================================================================
# 1) DEDICATED EXECUTOR FOR DARTS MODELS (uses the build_fn provided)
# ==============================================================================
def _darts_executor(
    build_fn,
    arch: str,
    train_kwargs: dict,
    data_inputs: dict,
    chunk_size: int,
    epochs: int,
    batch_size: int,
    patience: int,
    learning_rate: float,
) -> Dict[str, Any]:
    """
    Train `chunk_size` Darts models and average forecasts.
    Minimal logging: start/end + per-model start (INFO) and final summary.
    """
    import numpy as np

    logger = get_logger(__name__)
    with log_context(arch=arch, impl="darts", chunk_size=chunk_size):
        with phase(logger, "_darts_executor"):
            all_preds_test, all_preds_val = [], []

            for i in range(chunk_size):
                logger.info("training model=%d/%d", i + 1, chunk_size)
                # build_fn expected: model, history, pred_test, pred_val
                _model, _history, pred_test, pred_val = build_fn(
                    architecture_name=arch,
                    train_kwargs=train_kwargs,
                    data_inputs=data_inputs,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                    learning_rate=learning_rate,
                )
                all_preds_test.append(pred_test)
                all_preds_val.append(pred_val)

            final_pred_test = np.mean(all_preds_test, axis=0)
            final_pred_val  = np.mean(all_preds_val,  axis=0)

            out = {
                "successful_models": chunk_size,
                "pred_test": final_pred_test,
                "pred_val": final_pred_val,
                # Placeholders to align with TF outputs when keys are expected downstream
                "q_phys": None,
                "alpha": None,
            }
            logger.info("ensemble_averaged")
            return out




# # ==============================================================================
# 3) CLEAN DISPATCHER: signature matches your original _chunk_worker call
# ==============================================================================
def _chunk_worker(
    conn,
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
    learning_rate: float,
    max_retries: int,
    skip_on_failure: bool,
    agg_sigma: str = "approx",
):
    """
    Dispatches to Darts or TF executor.
    On failure, sends a single Exception embedding the child traceback.
    """
    logger = get_logger(__name__)
    with log_context(arch=arch, kind=kind, chunk_size=chunk_size):
        with phase(logger, "_chunk_worker", snapshots=bool(with_snapshots)):
            try:
                if arch.startswith("Darts_"):
                    logger.info("decision=executor impl=darts")
                    chunk_output = _darts_executor(
                        build_fn=build_fn,
                        arch=arch,
                        train_kwargs=train_kwargs,
                        data_inputs=data_inputs,
                        chunk_size=chunk_size,
                        epochs=epochs,
                        batch_size=batch_size,
                        patience=patience,
                        learning_rate=learning_rate,
                    )
                else:
                    logger.info("decision=executor impl=tf")
                    chunk_output = _tf_executor(
                        build_fn=build_fn,
                        arch=arch,
                        kind=kind,
                        train_kwargs=train_kwargs,
                        data_inputs=data_inputs,
                        chunk_size=chunk_size,
                        with_snapshots=with_snapshots,
                        epochs=epochs,
                        batch_size=batch_size,
                        patience=patience,
                        learning_rate=learning_rate,
                        max_retries=max_retries,
                        skip_on_failure=skip_on_failure,
                        agg_sigma=agg_sigma,
                    )

                conn.send(chunk_output)

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                wrapped = RuntimeError(
                    f"Chunk worker failed for arch='{arch}'.\n--- Child traceback ---\n{tb}"
                )
                wrapped.__cause__ = e
                logger.error("worker_exception_sent_upstream")
                conn.send(wrapped)
            finally:
                conn.close()



from typing import Dict, Any
from forecast_pipeline.logging_utils import get_logger, phase, log_context


def setup_training_environment(train_kwargs):
    """
    Sets up the environment for training, optionally enforcing determinism.
    """
    logger = get_logger(__name__)
    if ENABLE_DETERMINISM:
        logger.info("Determinism is enabled. Setting random seeds for reproducibility.")
        seed = train_kwargs.get("params", {}).get("seed", 42)

        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.config.experimental.enable_op_determinism()
    else:
        logger.info("Determinism is disabled. Training results may vary between runs.")


def _log_val_boundary_debug(logger, X_val, y_val, pred_val, scaler_target, target_idx_in_X=7, n_samples=3):
    # Usa os primeiros n_samples janelões da partição val
    B = min(n_samples, X_val.shape[0])
    # Último Q_s (escalado) do contexto
    q_hist_last_s = X_val[:B, -1, target_idx_in_X:target_idx_in_X+1]  # (B,1)
    oil_mean, oil_std = get_center_and_scale(scaler_target, as_tf=True, dtype=tf.float32)
    q_hist_last = (q_hist_last_s * oil_std + oil_mean).numpy().squeeze()  # físico
    q_pred0 = (pred_val[:B, 0]).squeeze()  # já em target scale → físico depois da inverse global
    # Aqui pred_val está em target scale (como legacy). Só logar o delta em target scale:
    delta0_target = (pred_val[:B, 0] - q_hist_last_s.squeeze()).astype(float)

    logger.warning(
        "val_boundary_debug B=%d hist_last_s[0:3]=%s pred0_s[0:3]=%s delta0_s[0:3]=%s",
        B, q_hist_last_s[:B].flatten()[:3], pred_val[:B,0].flatten()[:3], delta0_target[:B].flatten()[:3]
    )




# ---------------------------------------------------------------------
# Output schema helpers
# ---------------------------------------------------------------------

_OUT_SCHEMAS: Dict[int, Tuple[str, ...]] = {
    5: ("pred", "q_phys", "res", "sigma", "alpha"),
    4: ("pred", "q_phys", "res",            "alpha"),
    3: ("pred", "q_phys", "alpha"),
    1: ("pred",),
}


def _to_np(x):
    """Convert TensorFlow tensors to NumPy arrays, keeping NumPy inputs untouched."""
    return x.numpy() if isinstance(x, tf.Tensor) else np.asarray(x)


def _tuple_to_dict(out) -> Dict[str, np.ndarray]:
    """
    Map a model output tuple into a named dict using _OUT_SCHEMAS.

    Supports:
      (pred,),
      (pred, q_phys, alpha),
      (pred, q_phys, res, alpha),
      (pred, q_phys, res, sigma, alpha).
    """
    if not isinstance(out, tuple):
        out = (out,)

    try:
        schema = _OUT_SCHEMAS[len(out)]
    except KeyError:
        raise ValueError(f"Unsupported model output with {len(out)} branches")

    return {k: _to_np(out[i]) for i, k in enumerate(schema)}


def _online_mean(running: np.ndarray, new: np.ndarray, n: int) -> np.ndarray:
    """Streaming mean update: nth sample into running average."""
    return new.copy() if running is None else running + (new - running) / n


# ---------------------------------------------------------------------
# Prediction and accumulation helpers
# ---------------------------------------------------------------------

def _predict(
    model: tf.keras.Model,
    X,
    want_snaps_local: bool,
    agg_sigma: str = "approx",
) -> Tuple[Dict[str, np.ndarray], bool]:
    """
    Run model prediction, optionally aggregating snapshot weights.

    Returns:
        (outputs_dict, used_snapshots_flag)
    """
    snaps = getattr(model, "_snapshot_weights", None)
    use_snaps = bool(
        want_snaps_local
        and snaps is not None
        and len(snaps) > 0
    )

    if use_snaps:
        # Snapshot-based ensemble prediction
        cur = model.get_weights()
        outs = []
        for w in snaps:
            model.set_weights(w)
            outs.append(_tuple_to_dict(model(X, training=False)))
        model.set_weights(cur)

        keys = outs[0].keys()
        stk = {k: np.stack([d[k] for d in outs], axis=0) for k in keys}

        agg: Dict[str, np.ndarray] = {k: stk[k].mean(axis=0) for k in keys if k != "sigma"}
        if "sigma" in keys:
            if agg_sigma == "approx":
                mu_stack = stk["pred"]
                sigma_stack = stk["sigma"]
                agg["sigma"] = np.sqrt((sigma_stack ** 2).mean(axis=0) + mu_stack.var(axis=0))
            else:
                agg["sigma"] = stk["sigma"].mean(axis=0)

        return agg, True

    # Standard single-head prediction
    return _tuple_to_dict(model(X, training=False)), False


def _accumulate(
    src: Dict[str, np.ndarray],
    dest: Dict[str, np.ndarray],
    suffix: str,
    n_success: int,
) -> None:
    """
    Update running aggregates (online mean) for a given split.

    Notes:
      - 'q_phys' is only aggregated for validation, preserving legacy behaviour.
      - Other keys receive a split suffix (e.g. 'pred_test', 'pred_val').
    """
    for k, arr in src.items():
        if k == "q_phys" and suffix not in ("val",):
            continue
        key = k if k == "q_phys" else f"{k}_{suffix}"
        dest[key] = _online_mean(dest.get(key), arr, n_success)


from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------
# Latent adapter (PhysicsDecoder-based, full_sequence)
# ---------------------------------------------------------------------

def _maybe_apply_latent_adapter(
    model: tf.keras.Model,
    X,
    pred_dict: Dict[str, np.ndarray],
    split_name: str,
    *,
    arch: str,
    latent_mode: str,
    latent_cfg: Dict[str, Any],
    split_recon_lengths: Dict[str, int],
    scaler_X,
    scaler_target,
    extrap_models: Dict[str, tf.keras.Model],
    logger,
) -> Dict[str, np.ndarray]:
    """
    Optional PhysicsDecoder-based extrapolation (legacy latent adapter) com logs compactos.

    Logs:
      - build: 1 linha quando constrói o extrap_model (cache miss)
      - apply: 1 linha quando aplica, com métricas opcionais (debug ou drift grande)
      - skip: silencioso por default (para não poluir), exceto em debug_latent_adapter
    """
    # ------------------------------------------------------------
    # Guards (quiet by default)
    # ------------------------------------------------------------
    if pred_dict is None or "pred" not in pred_dict:
        return pred_dict
    if latent_mode != "full_sequence":
        return pred_dict

    base_split = split_name[:-5] if split_name.endswith("_left") else split_name
    if base_split not in ("val", "test"):
        return pred_dict
    if not isinstance(arch, str) or not arch.startswith("Seq2PIN"):
        return pred_dict

    steps = split_recon_lengths.get(base_split)
    if not steps or int(steps) <= 0:
        return pred_dict
    if scaler_X is None or scaler_target is None:
        return pred_dict

    import numpy as np

    lcfg = latent_cfg or {}

    # ------------------------------------------------------------
    # Logging policy knobs
    # ------------------------------------------------------------
    debug_latent = bool(
        lcfg.get("debug_latent_adapter", False)
        or lcfg.get("debug_latent", False)
        or lcfg.get("plot", False)
    )
    # só reclama se drift passar disso (em espaço do próprio "pred")
    warn_delta0 = lcfg.get("latent_warn_delta0", None)
    warn_deltamean = lcfg.get("latent_warn_deltamean", None)
    try:
        warn_delta0 = float(warn_delta0) if warn_delta0 is not None else None
    except Exception:
        warn_delta0 = None
    try:
        warn_deltamean = float(warn_deltamean) if warn_deltamean is not None else None
    except Exception:
        warn_deltamean = None

    # ------------------------------------------------------------
    # Config
    # ------------------------------------------------------------
    align_mode = (lcfg.get("align_on_overlap") or "off").strip().lower()  # off|shift|affine
    if align_mode not in {"off", "shift", "affine"}:
        align_mode = "off"

    baseline_pref = lcfg.get("baseline_key_preference", ["q_phys", "pred"])
    if not isinstance(baseline_pref, (list, tuple)) or len(baseline_pref) == 0:
        baseline_pref = ["q_phys", "pred"]

    eval_window_policy = (lcfg.get("eval_window_policy") or "first_only").strip().lower()
    if eval_window_policy not in {"first_only", "all_windows"}:
        if debug_latent:
            logger.warning(
                "latent_adapter_cfg_fix split=%s bad_eval_window_policy=%r -> first_only",
                str(split_name), eval_window_policy,
            )
        eval_window_policy = "first_only"

    # ------------------------------------------------------------
    # Build/cache extrap model
    # ------------------------------------------------------------
    key = f"{base_split}:{int(steps)}"
    if key not in extrap_models:
        from models.Seq2PIN_Trend import make_latent_extrapolation_model as _builder

        strat_cfg = {
            "strategy_name": lcfg.get("strategy_name", "arps"),
            "time_mode": "relative",
            "dt_scale": 1.0,
            "start_at_t0": False,
            "t_dt_M": 16,
            "feature_names": getattr(scaler_X, "feature_names_", None),
        }
        strat_cfg.update(lcfg.get("strategy_config") or {})

        extrap_models[key] = _builder(
            trained_model=model,
            scaler_X=scaler_X,
            scaler_target=scaler_target,
            strategy_config=strat_cfg,
            steps=int(steps),
            name=f"{arch}_latent_extrap_{base_split}_{int(steps)}",
        )

        logger.info(
            "latent_adapter_build mode=%s split=%s base_split=%s steps=%d model_name=%s",
            str(latent_mode), str(split_name), str(base_split), int(steps), str(extrap_models[key].name),
        )

    extrap_model = extrap_models[key]

    # ------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------
    try:
        # baseline key
        baseline_key = None
        for k in baseline_pref:
            if k in pred_dict:
                baseline_key = k
                break
        if baseline_key is None:
            baseline_key = "pred"

        pred_base = np.asarray(pred_dict[baseline_key])
        if pred_base.ndim == 1:
            pred_base = pred_base.reshape(1, -1)
        if pred_base.ndim != 2 or pred_base.size == 0:
            return pred_dict

        n_windows = int(pred_base.shape[0])

        # 1) Extrapola primeira janela vs todas
        source_policy = eval_window_policy
        if eval_window_policy == "first_only":
            X_first = X[:1]
            y_ext = extrap_model(X_first, training=False)  # (1, steps)
            y_ext_first = _to_np(y_ext)
            y_ext_np = np.repeat(y_ext_first, repeats=n_windows, axis=0)
        else:
            y_ext = extrap_model(X, training=False)
            y_ext_np = _to_np(y_ext)
            if y_ext_np.shape[0] != n_windows:
                # fallback defensivo (log só em debug)
                if debug_latent:
                    logger.warning(
                        "latent_adapter_broadcast split=%s got_windows=%d baseline_windows=%d",
                        str(split_name), int(y_ext_np.shape[0]), int(n_windows),
                    )
                if y_ext_np.shape[0] >= 1:
                    y_ext_np = np.repeat(y_ext_np[:1, :], repeats=n_windows, axis=0)
                else:
                    return pred_dict

        H_base = int(pred_base.shape[-1])
        H_ext = int(y_ext_np.shape[-1])
        H_common = int(min(H_base, H_ext))

        # 2) Métricas compactas (só se debug ou drift grande)
        delta0 = deltamean = None
        if H_common > 0:
            try:
                delta0 = float(np.mean(y_ext_np[:, 0] - pred_base[:, 0]))
                deltamean = float(np.mean(np.abs(y_ext_np[:, :H_common] - pred_base[:, :H_common])))
            except Exception:
                delta0 = deltamean = None

        drift_bad = False
        if (warn_delta0 is not None) and (delta0 is not None) and (abs(delta0) > warn_delta0):
            drift_bad = True
        if (warn_deltamean is not None) and (deltamean is not None) and (deltamean > warn_deltamean):
            drift_bad = True

        # 3) Alinhamento opcional (overlap)
        align_applied = "off"
        if H_common > 5 and align_mode in ("shift", "affine"):
            xb = y_ext_np[:, :H_common]
            yb = pred_base[:, :H_common]

            if align_mode == "shift":
                b_shift = (yb - xb).mean(axis=1, keepdims=True)  # (N,1)
                y_ext_np = y_ext_np + b_shift
                align_applied = "shift"

            else:
                eps = 1e-6
                Nw = xb.shape[0]
                ones = np.ones((H_common, 1), dtype=xb.dtype)
                lam = 1e-6
                for i in range(Nw):
                    X_ = np.concatenate([xb[i:i+1].T, ones], axis=1)  # (Hc,2)
                    y_ = yb[i:i+1].T
                    A = X_.T @ X_ + np.array([[lam, 0.0], [0.0, 0.0]], dtype=xb.dtype)
                    w = np.linalg.solve(A + eps * np.eye(2, dtype=xb.dtype), X_.T @ y_)
                    a = float(w[0, 0])
                    b = float(w[1, 0])
                    y_ext_np[i, :] = a * y_ext_np[i, :] + b
                align_applied = "affine"

        # 4) Log compacto (sempre 1 linha quando aplicado)
        #    - métricas só entram se debug ou drift suspeito
        msg = (
            "latent_adapter_apply split=%s base_split=%s policy=%s align=%s "
            "baseline=%s N=%d H_base=%d H_ext=%d H_target=%d"
        )
        if debug_latent or drift_bad:
            msg += " delta0=%.6g deltamean=%.6g"

        if (debug_latent or drift_bad) and (delta0 is not None) and (deltamean is not None):
            logger.info(
                msg,
                str(split_name),
                str(base_split),
                str(source_policy),
                str(align_applied),
                str(baseline_key),
                int(n_windows),
                int(H_base),
                int(H_ext),
                int(steps),
                float(delta0),
                float(deltamean),
            )
        else:
            logger.info(
                msg,
                str(split_name),
                str(base_split),
                str(source_policy),
                str(align_applied),
                str(baseline_key),
                int(n_windows),
                int(H_base),
                int(H_ext),
                int(steps),
            )

        new_pred = dict(pred_dict)
        new_pred["pred_phys_extrap"] = y_ext_np
        new_pred["pred"] = y_ext_np
        new_pred.setdefault("meta", {})
        if isinstance(new_pred["meta"], dict):
            new_pred["meta"]["latent_adapter"] = {
                "mode": str(latent_mode),
                "policy": str(source_policy),
                "align": str(align_applied),
                "baseline": str(baseline_key),
                "n_windows": int(n_windows),
                "H_base": int(H_base),
                "H_ext": int(H_ext),
                "H_target": int(steps),
                "delta0": (float(delta0) if delta0 is not None else None),
                "deltamean": (float(deltamean) if deltamean is not None else None),
            }

        return new_pred

    except Exception as ex:
        # erro real: vale logar
        logger.exception(
            "latent_adapter_failed split=%s base_split=%s error=%s",
            str(split_name),
            str(base_split),
            f"{type(ex).__name__}: {ex}",
        )
        return pred_dict





from forecast_pipeline.arps_offline import (
    _fit_arps_core_from_pinn,
    _apply_arps_from_pinn,
    _maybe_coupling_offline_analytic,
    compute_history_q0_phys,
)


# ---------------------------------------------------------------------
# Offline analytic extrapolation (exponential / Arps b=0)
# ---------------------------------------------------------------------
def _maybe_apply_offline_analytic(
    *,
    split_name: str,
    pred_dict: Dict[str, Any],
    scaler_y,
    split_recon_lengths: Dict[str, Any],
    latent_cfg: Dict[str, Any],
    logger,
) -> Dict[str, Any]:
    """
    Offline-analytic ARPS override (simplificado).

    Regras (strict):
      - q0 vem SOMENTE do histórico real:
          pred_dict['q0_anchor_phys'] (top-level) OU pred_dict['anchor']['q0_phys']
      - sem fallback para PINN-head (nunca ancorar na predição da PINN)
      - se não houver anchor válido -> retorna pred_dict unchanged
    """
    import numpy as np
    from typing import Optional, Dict, Any

    # ------------------------- log_utils (compact blocks) -------------------------
    try:
        from common.log_utils import (
            log_kv_block,
            effective_log_width,
            is_compact_logging,
        )
    except Exception:
        def log_kv_block(title, kv, level=None, width=100, key_pad=None):
            logger.info("%s %s", title, dict(kv or {}))
        def effective_log_width(cfg=None, fallback=100): return fallback
        def is_compact_logging(cfg=None): return True

    W = effective_log_width(None, fallback=100)
    compact = is_compact_logging(None)

    def _emit(title: str, kv: Dict[str, Any]) -> None:
        # Em compact mode: bloco. Em verbose: uma linha curta.
        if compact:
            log_kv_block(title, kv, width=W)
        else:
            # linha curta sem “log verboso”
            base = ", ".join([f"{k}={v}" for k, v in kv.items() if v is not None])
            logger.info("%s | %s", title, base)

    def _shape(x: Any) -> Any:
        try:
            s = np.asarray(x).shape
            return tuple(s)
        except Exception:
            try:
                return tuple(getattr(x, "shape", None) or ())
            except Exception:
                return None

    def _to_float_or_none(x) -> Optional[float]:
        try:
            if x is None:
                return None
            v = float(x)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    def _mode_of(cfg: Dict[str, Any]) -> str:
        return str(cfg.get("mode", cfg.get("latent_mode", "off"))).strip().lower()

    # ------------------------- Guards (silenciosos, mas explicáveis) -------------------------
    if not isinstance(pred_dict, dict) or "pred" not in pred_dict:
        # não é um split “previsível”
        return pred_dict

    cfg = dict(latent_cfg or {})
    mode = _mode_of(cfg)
    if mode != "offline_analytic":
        return pred_dict

    base_split = split_name[:-5] if str(split_name).endswith("_left") else split_name
    if base_split not in ("val", "test"):
        return pred_dict

    # H_target
    try:
        H_target = int((split_recon_lengths or {}).get(base_split, 0))
    except Exception:
        H_target = 0
    if H_target <= 0:
        _emit(
            f"Offline analytic — {split_name}",
            {"status": "skip", "reason": "H_target<=0", "base_split": base_split, "H_target": H_target},
        )
        return pred_dict

    # pred array
    try:
        pred = np.asarray(pred_dict["pred"], dtype=float)
    except Exception:
        _emit(
            f"Offline analytic — {split_name}",
            {"status": "skip", "reason": "pred_not_array", "base_split": base_split},
        )
        return pred_dict

    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if pred.ndim != 2 or pred.size == 0:
        _emit(
            f"Offline analytic — {split_name}",
            {"status": "skip", "reason": "pred_not_2d_or_empty", "pred_shape": _shape(pred)},
        )
        return pred_dict

    N, H_train = pred.shape
    debug = bool(cfg.get("debug_arps", False))

    # ---- resolve q0 anchor (PHYSICAL) sem fallback ----
    q0_anchor_phys = _to_float_or_none(pred_dict.get("q0_anchor_phys"))
    q0_source = "top.q0_anchor_phys" if q0_anchor_phys is not None else "none"

    if q0_anchor_phys is None:
        anc = pred_dict.get("anchor")
        if isinstance(anc, dict):
            q0_anchor_phys = _to_float_or_none(anc.get("q0_phys"))
            if q0_anchor_phys is not None:
                q0_source = "anchor.q0_phys"

    # min q0
    min_q0 = cfg.get("history_anchor_min_q0_phys", cfg.get("q0_min_phys", 1e-6))
    try:
        min_q0 = float(min_q0)
    except Exception:
        min_q0 = 1e-6
    min_q0 = max(0.0, float(min_q0))

    if q0_anchor_phys is None or (q0_anchor_phys <= min_q0):
        # aqui o log é MUITO útil pro bug: mas não precisa verbosidade
        _emit(
            f"Offline analytic — {split_name}",
            {
                "status": "skip",
                "reason": "missing_or_tiny_history_anchor",
                "base_split": base_split,
                "pred_shape": (N, H_train),
                "H_target": H_target,
                "q0": q0_anchor_phys,
                "min_q0": min_q0,
                "q0_source": q0_source,
            },
        )
        return pred_dict

    # ---- knobs mínimos ----
    fit_window = cfg.get("arps_fit_window", min(80, H_train))
    try:
        fit_window = int(fit_window)
    except Exception:
        fit_window = min(80, H_train)
    fit_window = max(1, min(fit_window, H_train))

    fit_region = str(cfg.get("analytic_fit_region", "head")).strip().lower()
    anchor_kind = str(cfg.get("analytic_anchor_kind", "median")).strip().lower()
    anchor_window = cfg.get("analytic_anchor_window", cfg.get("arps_anchor_window", 10))
    try:
        anchor_window = int(anchor_window)
    except Exception:
        anchor_window = 10
    anchor_window = max(1, anchor_window)

    force_monotonic = bool(cfg.get("analytic_force_monotonic", True))

    max_decline = cfg.get("analytic_max_decline_per_step", None)
    try:
        max_decline = float(max_decline) if max_decline is not None else None
    except Exception:
        max_decline = None

    override_pred = bool(cfg.get("analytic_override_pred", True))

    # b policies
    use_only_first_window = bool(cfg.get("analytic_use_only_first_window", True))
    b_fit_policy = str(cfg.get("b_fit_policy", "first_k" if use_only_first_window else "all")).strip().lower()
    b_reducer = str(cfg.get("b_reducer", "median")).strip().lower()
    b_fit_k = cfg.get("b_fit_k", 1 if use_only_first_window else int(cfg.get("default_b_fit_k", 6)))

    def _parse_k(x, n_max: int) -> Optional[int]:
        try:
            if x is None:
                return None
            if isinstance(x, str) and x.strip().lower() in ("", "none", "null", "auto", "all"):
                return None
            v = int(float(x))
            if v <= 0:
                return None
            return max(1, min(v, int(n_max)))
        except Exception:
            return None

    b_fit_k_parsed = _parse_k(b_fit_k, n_max=N)

    # Snapshot curto antes do fit (alta utilidade, zero verbosidade)
    _emit(
        f"Offline analytic — {split_name}",
        {
            "status": "enter",
            "base_split": base_split,
            "pred_shape": (N, H_train),
            "H_target": H_target,
            "q0": float(q0_anchor_phys),
            "q0_source": q0_source,
            "fit_region": fit_region,
            "fit_window": fit_window,
            "anchor_kind": anchor_kind,
            "anchor_window": anchor_window,
            "b_fit_policy": b_fit_policy,
            "b_fit_k": (b_fit_k_parsed if b_fit_k_parsed is not None else "all"),
            "b_reducer": b_reducer,
            "override_pred": override_pred,
        },
    )

    # keep a copy of the PINN ribbon for coupling/debug (BEFORE overriding pred)
    pred_pinn_coupling = np.array(pred, copy=True)

    try:
        y_analytic_scaled, params = _fit_arps_core_from_pinn(
            split_name=str(split_name),
            base_split=str(base_split),
            y_pinn_scaled=pred,
            scaler_y=scaler_y,
            H_target=int(H_target),
            fit_window=int(fit_window),
            force_monotonic=bool(force_monotonic),
            max_decline_per_step=max_decline,
            fit_region=str(fit_region),
            anchor_kind=str(anchor_kind),
            anchor_window=int(anchor_window),
            logger=logger,
            b_fit_policy=str(b_fit_policy),
            b_fit_k=b_fit_k_parsed,
            b_reducer=str(b_reducer),
            # STRICT: q0 vem do histórico real
            q0_override_phys=float(q0_anchor_phys),
            min_q0_phys=float(min_q0),
        )
    except Exception as ex:
        _emit(
            f"Offline analytic — {split_name}",
            {
                "status": "skip",
                "reason": "analytic_failed",
                "err": f"{type(ex).__name__}: {ex}",
                "base_split": base_split,
            },
        )
        if debug:
            try:
                logger.exception(
                    "offline_analytic_skip split=%s base_split=%s reason=analytic_failed error=%s",
                    split_name, base_split, str(ex),
                )
            except Exception:
                pass
        return pred_dict

    out = dict(pred_dict)
    out["pred_analytic"] = y_analytic_scaled
    out["pred_pinn_coupling"] = pred_pinn_coupling
    out["analytic_params"] = dict(params or {})
    out["analytic_params"]["q0_source_intent"] = "history_anchor_only"
    out["analytic_params"]["q0_override_phys_input"] = float(q0_anchor_phys)

    # mantém anchor em dict (compat agregador)
    out.setdefault("anchor", {})
    if isinstance(out["anchor"], dict):
        out["anchor"]["q0_phys"] = float(q0_anchor_phys)
        if isinstance(pred_dict.get("anchor"), dict) and pred_dict["anchor"].get("q0_meta") is not None:
            out["anchor"]["q0_meta"] = dict(pred_dict["anchor"].get("q0_meta") or {})

    if override_pred:
        out["pred"] = y_analytic_scaled

    # Snapshot curto pós-fit (b / dispersão / shape)
    p = dict(params or {})
    b_star = _to_float_or_none(p.get("b"))
    b_std = _to_float_or_none(p.get("b_std"))
    b_std_raw = _to_float_or_none(p.get("b_std_raw"))
    n_b = None
    try:
        bc = p.get("b_candidates", None)
        n_b = (len(bc) if isinstance(bc, list) else None)
    except Exception:
        n_b = None

    _emit(
        f"Offline analytic — {split_name}",
        {
            "status": ("applied" if override_pred else "computed_only"),
            "pred_out_shape": _shape(out.get("pred")),
            "pred_analytic_shape": _shape(out.get("pred_analytic")),
            "b": b_star,
            "b_std": b_std,
            "b_std_raw": b_std_raw,
            "b_candidates_n": n_b,
            "q0": float(q0_anchor_phys),
        },
    )

    return out





def _tf_executor(
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
    learning_rate: float,
    max_retries: int,
    skip_on_failure: bool,
    agg_sigma: str = "approx",
) -> dict:
    """
    TensorFlow executor with optional snapshots, LEFT predictions, and latent-aware
    post-processing (full_sequence adapter or offline_analytic ARPS-like extrapolation).
    """
    import gc
    import traceback
    from typing import Any, Dict, Optional, Tuple

    import numpy as np
    import tensorflow as tf

    logger = get_logger(__name__)
    setup_training_environment(train_kwargs)

    # ------------------------- NEW: log_utils (compact + blocks) -------------------------
    try:
        from common.log_utils import (
            stage_banner,
            log_kv_block,
            log_block,
            effective_log_width,
            is_compact_logging,
            ok,
            warn,
            info,
            err,
        )
    except Exception:
        # fallback mínimo (não quebra)
        def stage_banner(n, title, subtitle="", width=92): logger.info(f"[Stage {n}] {title} {subtitle}")
        def log_kv_block(title, kv, level=None, width=100, key_pad=None): logger.info("%s %s", title, dict(kv or {}))
        def log_block(title, lines, level=None, width=100): logger.info("%s\n%s", title, "\n".join(map(str, lines or [])))
        def effective_log_width(cfg=None, fallback=100): return fallback
        def is_compact_logging(cfg=None): return True
        def ok(m, *a, **k): logger.info("✅ " + str(m), *a, **k)
        def warn(m, *a, **k): logger.warning("⚠️  " + str(m), *a, **k)
        def info(m, *a, **k): logger.info("ℹ️  " + str(m), *a, **k)
        def err(m, *a, **k): logger.error("❌ " + str(m), *a, **k)

    W = effective_log_width(None, fallback=100)
    compact = is_compact_logging(None)

    def _shape(x: Any) -> Any:
        try:
            return tuple(getattr(x, "shape", None) or ())
        except Exception:
            return None

    def _scaler_sig(s: Any) -> str:
        if s is None:
            return "None"
        nfi = getattr(s, "n_features_in_", None)
        cls = type(s).__name__
        return f"{cls}(n_features_in_={nfi})"

    def _safe_float(x: Any) -> Optional[float]:
        try:
            v = float(x)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    def _rank_summary_if_possible(*, split: str, outs: dict) -> Optional[Dict[str, Any]]:
        """
        Usa rank_position_summary se existir e se houver members_scaled + pred.
        Não quebra se não existir.
        """
        if not isinstance(outs, dict):
            return None
        try:
            # preferir função local/importada (você já adicionou no arps_offline)
            try:
                from forecast_pipeline.arps_offline import rank_position_summary  # type: ignore
            except Exception:
                rank_position_summary = None  # type: ignore

            if rank_position_summary is None:
                return None

            # tenta achar members_scaled (usa helpers já existentes mais abaixo)
            ms = _pick_members_scaled_anywhere(outs, "val" if split == "val" else "test")
            if ms is None:
                return None
            p1d = _reduce_pred_to_1d(outs.get("pred"))
            if p1d is None:
                return None

            s = rank_position_summary(ms, p1d)
            if not isinstance(s, dict):
                return None
            # dá um “sinal” pequeno e legível
            return {
                "T": s.get("T"),
                "rank_med": s.get("rank_med"),
                "rank_q10": s.get("rank_q10"),
                "rank_q90": s.get("rank_q90"),
                "rank_bad%": s.get("rank_bad%"),
            }
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _detect_hpo(train_kwargs_: dict, data_inputs_: dict) -> Tuple[bool, str]:
        run_kind_ = str(train_kwargs_.get("run_kind", data_inputs_.get("run_kind", ""))).lower()
        is_hpo_ = bool(
            train_kwargs_.get("is_hpo")
            or data_inputs_.get("is_hpo")
            or run_kind_ in {"hpo", "hyperopt", "optuna"}
        )
        return is_hpo_, run_kind_

    def _normalize_latent_cfg(*, is_hpo_: bool, run_kind_: str, upstream: dict) -> dict:
        if is_hpo_:
            logger.info(
                "latent_ctx: HPO run detected (run_kind=%r). Disabling offline analytic and ARPS coupling.",
                run_kind_,
            )
            return {"mode": "off"}

        cfg = dict(upstream or {})
        mode_raw = cfg.get("mode", cfg.get("latent_mode", None))
        mode = str(mode_raw).strip().lower() if mode_raw is not None else "off"

        if not mode:
            mode = "off"
            cfg["mode"] = mode
        elif mode not in {"off", "full_sequence", "offline_analytic"}:
            logger.warning("latent_ctx: unknown upstream mode=%r; falling back to 'off'.", mode)
            mode = "off"
            cfg["mode"] = mode

        if mode == "offline_analytic":
            # -------------------------
            # Safe defaults (DO NOT auto-enable spaghetti/sampling)
            # Presets are the single entry-point that should enable advanced behavior.
            # -------------------------
            cfg.setdefault("arps_fit_window", 80)
            cfg.setdefault("analytic_fit_region", "head")
            cfg.setdefault("analytic_anchor_kind", "median")
            cfg.setdefault("analytic_anchor_window", 10)
            cfg.setdefault("analytic_use_only_first_window", True)

            cfg.setdefault("analytic_override_pred", True)
            cfg.setdefault("analytic_force_monotonic", True)
            cfg.setdefault("analytic_max_decline_per_step", 0.15)

            # Coupling defaults (safe)
            cfg.setdefault("arps_coupling_mode", "val_only")
            cfg.setdefault("arps_coupling_fit_window", 90)
            cfg.setdefault("arps_anchor_window", 15)

            # IMPORTANT: these must NOT be enabled implicitly.
            # They should be set ONLY by a preset like ARPS_ENSEMBLE_SPAGHETTI.
            cfg.setdefault("coupling_spaghetti", None)
            cfg.setdefault("coupling_spaghetti_k", 0)

            # NEW: theta sampling around θ̂ (disabled by default)
            cfg.setdefault("coupling_theta_sampling", False)
            cfg.setdefault("coupling_theta_sampling_dist", "normal")
            cfg.setdefault("coupling_theta_sampling_sigma", 0.003)       # absolute
            cfg.setdefault("coupling_theta_sampling_rel_sigma", None)    # optional alt
            cfg.setdefault("coupling_theta_sampling_clip_min", None)
            cfg.setdefault("coupling_theta_sampling_clip_max", None)

            # Optional trimming defaults (safe no-op unless enabled in preset)
            cfg.setdefault("traj_outlier_filter", "none")
            cfg.setdefault("traj_outlier_trim_pct", 0.10)
            cfg.setdefault("traj_outlier_min_keep", 20)

        return cfg

    def _cleanup_session(*, reason: str) -> None:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()
        logger.debug("cleanup_session reason=%s", reason)

    def _get_scaler_target(train_kwargs_: dict, data_inputs_: dict):
        return train_kwargs_.get("scaler_target") or data_inputs_.get("scaler_target")

    def _get_sidecar_left(scaler_target_) -> Tuple[bool, bool, Optional[Any], Optional[Any]]:
        sidecar_ = getattr(scaler_target_, "_split_ctx", None) if scaler_target_ is not None else None
        has_val_ = bool(sidecar_ and sidecar_.get("X_val_left_scaled") is not None)
        has_test_ = bool(sidecar_ and sidecar_.get("X_test_left_scaled") is not None)
        x_val_left_ = sidecar_.get("X_val_left_scaled") if has_val_ else None
        x_test_left_ = sidecar_.get("X_test_left_scaled") if has_test_ else None
        return has_val_, has_test_, x_val_left_, x_test_left_

    def _apply_postprocessing(
        *,
        model,
        X,
        split_name: str,
        pred_dict: Dict[str, Any],
        used_snaps: bool,
    ) -> Dict[str, Any]:
        out_ = _maybe_apply_latent_adapter(
            model=model,
            X=X,
            pred_dict=pred_dict,
            split_name=split_name,
            arch=arch,
            latent_mode=latent_mode,
            latent_cfg=latent_cfg,
            split_recon_lengths=split_recon_lengths,
            scaler_X=scaler_X,
            scaler_target=scaler_target,
            extrap_models=extrap_models,
            logger=logger,
        )

        # ------------------------------------------------------------
        # 1) Attach history anchor (q0) from X[-1] in PHYSICAL units
        # ------------------------------------------------------------
        def _norm_str(x, default: str) -> str:
            s = str(x).strip().lower() if x is not None else default
            return s or default

        try:
            cfg = dict(latent_cfg or {})
            mode = _norm_str(cfg.get("mode", cfg.get("latent_mode", "off")), "off")
            base_split = split_name[:-5] if str(split_name).endswith("_left") else split_name

            if mode == "offline_analytic" and base_split in ("val", "test") and isinstance(out_, dict):
                if not (isinstance(out_.get("anchor"), dict) and out_["anchor"].get("q0_phys") is not None):
                    debug = bool(cfg.get("debug_arps", False))
                    expose_top_level = bool(cfg.get("history_anchor_expose_top_level", False))

                    w = cfg.get("history_anchor_window", cfg.get("q0_anchor_window", cfg.get("analytic_anchor_window", 10)))
                    try:
                        w = int(w)
                    except Exception:
                        w = 10
                    w = max(1, w)

                    kind_local = _norm_str(cfg.get("history_anchor_kind", cfg.get("q0_anchor_kind", "median")), "median")

                    min_q0 = cfg.get("history_anchor_min_q0_phys", cfg.get("q0_min_phys", 1e-6))
                    try:
                        min_q0 = float(min_q0)
                    except Exception:
                        min_q0 = 1e-6
                    min_q0 = max(0.0, float(min_q0))

                    x_in_scaler_space = cfg.get("history_anchor_x_in_scaler_space", None)
                    if x_in_scaler_space is not None:
                        x_in_scaler_space = bool(x_in_scaler_space)

                    ch = cfg.get("history_anchor_channel", -1)
                    try:
                        ch = int(ch)
                    except Exception:
                        ch = -1

                    q0_phys, q0_meta = compute_history_q0_phys(
                        X_any=X,
                        scaler_X=scaler_X,
                        scaler_target=scaler_target,
                        window=w,
                        kind=kind_local,
                        min_q0_phys=min_q0,
                        channel=ch,
                        logger=logger,
                        debug=debug,
                        x_in_scaler_space=x_in_scaler_space,
                    )

                    if q0_phys is not None:
                        out_.setdefault("anchor", {})
                        if isinstance(out_["anchor"], dict):
                            out_["anchor"]["q0_phys"] = float(q0_phys)
                            out_["anchor"]["q0_meta"] = {
                                **(q0_meta or {}),
                                "source": "history_context_tail",
                                "split": str(split_name),
                                "base_split": str(base_split),
                                "units": "physical",
                            }

                        if expose_top_level:
                            out_["q0_anchor_phys"] = float(q0_phys)
                            out_["q0_anchor_meta"] = dict(out_["anchor"].get("q0_meta") or {})

                    elif debug:
                        logger.info(
                            "history_anchor_skip split=%s base_split=%s reason=%s meta=%s",
                            split_name, base_split, str((q0_meta or {}).get("reason")), str(q0_meta),
                        )
        except Exception as ex:
            try:
                logger.warning("history_anchor_failed split=%s error=%s", split_name, str(ex))
            except Exception:
                pass

        # ------------------------------------------------------------
        # 2) Apply offline analytic
        # ------------------------------------------------------------
        try:
            out_ = _maybe_apply_offline_analytic(
                split_name=split_name,
                pred_dict=out_,
                scaler_y=scaler_target,
                split_recon_lengths=split_recon_lengths,
                latent_cfg=latent_cfg,
                logger=logger,
            )
        except Exception as ex:
            try:
                logger.warning("offline_analytic_failed split=%s error=%s", split_name, str(ex))
            except Exception:
                pass

        return out_

    def _predict_split(model, X, split_name: str) -> Dict[str, Any]:
        pred_dict, used_snaps_local = _predict(
            model,
            X,
            want_snaps_local=want_snaps,
            agg_sigma=agg_sigma,
        )
        return _apply_postprocessing(
            model=model,
            X=X,
            split_name=split_name,
            pred_dict=pred_dict,
            used_snaps=bool(used_snaps_local),
        )

    # ---------------------------------------------------------------------
    # Spaghetti capture (visual-only; SCALE-SAFE)
    # ---------------------------------------------------------------------
    captured_val_members_scaled = None
    captured_test_members_scaled = None
    captured_val_members_meta = None
    captured_test_members_meta = None

    def _as_meta_dict(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(d, dict):
            return {}
        m = d.get("meta")
        return m if isinstance(m, dict) else {}

    def _first_not_none(*xs):
        for x in xs:
            if x is not None:
                return x
        return None

    def _to_2d_float(a: Any) -> Optional[np.ndarray]:
        if a is None:
            return None
        try:
            arr = np.asarray(a, dtype=float)
        except Exception:
            return None
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            return None
        return arr

    def _pick_members_scaled_anywhere(d: Optional[Dict[str, Any]], split: str) -> Optional[np.ndarray]:
        if not isinstance(d, dict):
            return None
        meta = _as_meta_dict(d)

        if split == "val":
            cand = _first_not_none(
                d.get("integrated_view_val_members_scaled"),
                d.get("pred_val_members_scaled"),
                d.get("pred_members_val_scaled"),
                d.get("pred_members_scaled"),
                meta.get("integrated_view_val_members_scaled"),
                meta.get("pred_val_members_scaled"),
                meta.get("pred_members_val_scaled"),
                meta.get("pred_members_scaled"),
            )
            return _to_2d_float(cand)

        cand = _first_not_none(
            d.get("integrated_view_test_members_scaled"),
            d.get("pred_test_members_scaled"),
            d.get("pred_members_scaled"),
            meta.get("integrated_view_test_members_scaled"),
            meta.get("pred_test_members_scaled"),
            meta.get("pred_members_scaled"),
        )
        return _to_2d_float(cand)

    def _pick_members_meta_anywhere(d: Optional[Dict[str, Any]], split: str) -> Dict[str, Any]:
        if not isinstance(d, dict):
            return {}
        meta = _as_meta_dict(d)

        if split == "val":
            m = _first_not_none(
                d.get("integrated_view_val_members_meta"),
                d.get("pred_val_members_meta"),
                d.get("pred_members_val_meta"),
                d.get("pred_members_meta"),
                meta.get("integrated_view_val_members_meta"),
                meta.get("pred_val_members_meta"),
                meta.get("pred_members_val_meta"),
                meta.get("pred_members_meta"),
            )
            return dict(m or {})

        m = _first_not_none(
            d.get("integrated_view_test_members_meta"),
            d.get("pred_test_members_meta"),
            d.get("pred_members_meta"),
            meta.get("integrated_view_test_members_meta"),
            meta.get("pred_test_members_meta"),
            meta.get("pred_members_meta"),
        )
        return dict(m or {})

    def _capture_members_once(*, outs_val_: dict, outs_test_: dict) -> None:
        nonlocal captured_val_members_scaled, captured_test_members_scaled, captured_val_members_meta, captured_test_members_meta

        if is_hpo:
            return

        if captured_val_members_scaled is None:
            mv = _pick_members_scaled_anywhere(outs_val_, "val")
            if mv is not None:
                captured_val_members_scaled = mv.copy()
                captured_val_members_meta = _pick_members_meta_anywhere(outs_val_, "val")
                if compact:
                    log_kv_block(
                        "Spaghetti capture — VAL",
                        {"captured": "yes", "shape": tuple(captured_val_members_scaled.shape), "meta_keys": sorted(list((captured_val_members_meta or {}).keys()))[:12]},
                        width=W,
                    )
                else:
                    logger.info("spaghetti_capture val=yes shape=%s", tuple(captured_val_members_scaled.shape))

        if captured_test_members_scaled is None:
            mt = _pick_members_scaled_anywhere(outs_test_, "test")
            if mt is not None:
                captured_test_members_scaled = mt.copy()
                captured_test_members_meta = _pick_members_meta_anywhere(outs_test_, "test")
                if compact:
                    log_kv_block(
                        "Spaghetti capture — TEST",
                        {"captured": "yes", "shape": tuple(captured_test_members_scaled.shape), "meta_keys": sorted(list((captured_test_members_meta or {}).keys()))[:12]},
                        width=W,
                    )
                else:
                    logger.info("spaghetti_capture test=yes shape=%s", tuple(captured_test_members_scaled.shape))

    def _reduce_pred_to_1d(pred_any: Any) -> Optional[np.ndarray]:
        if pred_any is None:
            return None
        arr = np.asarray(pred_any, dtype=float)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            return arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
        return None

    def _synthesize_val_members_if_missing(*, outs_val_: dict, outs_test_: dict) -> None:
        if is_hpo:
            return
        if latent_mode == "offline_analytic":
            return
        if not isinstance(outs_val_, dict) or not isinstance(outs_test_, dict):
            return
        if _pick_members_scaled_anywhere(outs_val_, "val") is not None:
            return

        test_members = _pick_members_scaled_anywhere(outs_test_, "test")
        if test_members is None:
            return

        base_val = _reduce_pred_to_1d(outs_val_.get("pred"))
        if base_val is None:
            return

        L = int(split_recon_lengths.get("val", base_val.shape[-1]))
        L = max(1, min(L, base_val.shape[-1]))

        K = int(latent_cfg.get("coupling_spaghetti_k", test_members.shape[0]))
        K = max(1, min(K, test_members.shape[0]))

        members = test_members[:K, :L].copy()
        try:
            members[:, 0] = float(base_val[0])
        except Exception:
            pass

        outs_val_["pred_members_scaled"] = members
        outs_val_["pred_members_meta"] = {"source": "synth_from_test_prefix_aligned", "K": int(K), "L": int(L)}
        if compact:
            log_kv_block("Spaghetti synth — VAL (legacy fallback)", {"shape": tuple(members.shape), "K": K, "L": L}, width=W)
        else:
            logger.info("spaghetti_synth val=yes shape=%s", tuple(members.shape))

    # ---------------------------------------------------------------------
    # Detect HPO / normalize latent config
    # ---------------------------------------------------------------------
    is_hpo, run_kind = _detect_hpo(train_kwargs, data_inputs)
    latent_cfg_upstream = dict(train_kwargs.get("latent_cfg") or {})
    latent_cfg = _normalize_latent_cfg(is_hpo_=is_hpo, run_kind_=run_kind, upstream=latent_cfg_upstream)
    train_kwargs["latent_cfg"] = latent_cfg
    latent_mode = str(latent_cfg.get("mode", latent_cfg.get("latent_mode", "off"))).strip().lower()

    # ---------------------------------------------------------------------
    # Context / scalers / split lengths
    # ---------------------------------------------------------------------
    split_recon_lengths = train_kwargs.get("split_recon_lengths") or {}
    scaler_X = train_kwargs.get("scaler_X")
    scaler_target = _get_scaler_target(train_kwargs, data_inputs)

    # ------------------------- NEW: Stage 0/1 banners -------------------------
    stage_banner(0, "_tf_executor", f"arch={arch} kind={kind} chunk_size={chunk_size}", width=W)
    log_kv_block(
        "Run context",
        {
            "run_kind": run_kind,
            "is_hpo": is_hpo,
            "want_snaps": bool(True if with_snapshots is None else bool(with_snapshots)),
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "learning_rate": learning_rate,
            "max_retries": max_retries,
            "skip_on_failure": skip_on_failure,
        },
        width=W,
    )

    stage_banner(1, "Latent config (effective)", f"mode={latent_mode}", width=W)
    # mostra só o que importa pro bug (offline_analytic/coupling/spaghetti/anchor)
    if latent_mode == "offline_analytic":
        log_kv_block(
            "Offline-analytic knobs",
            {
                "arps_coupling_mode": latent_cfg.get("arps_coupling_mode"),
                "coupling_spaghetti": latent_cfg.get("coupling_spaghetti"),
                "coupling_spaghetti_k": latent_cfg.get("coupling_spaghetti_k"),
                "coupling_spaghetti_reducer": latent_cfg.get("coupling_spaghetti_reducer", latent_cfg.get("coupling_spaghetti_agg")),
                "traj_outlier_filter": latent_cfg.get("traj_outlier_filter"),
                "traj_outlier_trim_pct": latent_cfg.get("traj_outlier_trim_pct"),
                "traj_outlier_min_keep": latent_cfg.get("traj_outlier_min_keep"),
                "analytic_override_pred": latent_cfg.get("analytic_override_pred"),
                "debug_arps": latent_cfg.get("debug_arps"),
                "history_anchor_x_in_scaler_space": latent_cfg.get("history_anchor_x_in_scaler_space"),
                "history_anchor_channel": latent_cfg.get("history_anchor_channel"),
                "history_anchor_window": latent_cfg.get("history_anchor_window", latent_cfg.get("analytic_anchor_window")),
                "history_anchor_kind": latent_cfg.get("history_anchor_kind", latent_cfg.get("analytic_anchor_kind")),
                "history_anchor_min_q0_phys": latent_cfg.get("history_anchor_min_q0_phys", latent_cfg.get("q0_min_phys")),
            },
            width=W,
        )
    else:
        log_kv_block("Latent knobs", {"mode": latent_mode}, width=W)

    if split_recon_lengths:
        log_kv_block(
            "Split recon lengths",
            {"val": split_recon_lengths.get("val"), "test": split_recon_lengths.get("test")},
            width=W,
        )
    else:
        info("latent_ctx split_recon_lengths=None")

    log_kv_block(
        "Scalers",
        {"scaler_X": _scaler_sig(scaler_X), "scaler_target": _scaler_sig(scaler_target)},
        width=W,
    )

    extrap_models: Dict[str, tf.keras.Model] = {}

    # ---------------------------------------------------------------------
    # Inputs
    # ---------------------------------------------------------------------
    X_test = data_inputs["X_test"]
    X_val = data_inputs["X_val"]

    has_left_val, has_left_test, X_val_left, X_test_left = _get_sidecar_left(scaler_target)

    stage_banner(2, "Inputs", "shapes + left presence", width=W)
    log_kv_block(
        "Data shapes",
        {"X_val": _shape(X_val), "X_test": _shape(X_test), "X_val_left": _shape(X_val_left), "X_test_left": _shape(X_test_left)},
        width=W,
    )
    log_kv_block(
        "Left presence",
        {"val_left": ("yes" if has_left_val else "no"), "test_left": ("yes" if has_left_test else "no")},
        width=W,
    )

    want_snaps = True if with_snapshots is None else bool(with_snapshots)

    # ---------------------------------------------------------------------
    # Accumulators
    # ---------------------------------------------------------------------
    running_test: Dict[str, Any] = {}
    running_val: Dict[str, Any] = {}
    running_alpha_sum = 0.0
    models_with_alpha = 0
    successful = 0

    with log_context(arch=arch, impl="tf", chunk_size=chunk_size, snaps="want" if want_snaps else "no"):
        with phase(logger, "_tf_executor"):
            try:
                stage_banner(3, "Training loop", "fit → predict → (optional) coupling → accumulate", width=W)

                for idx in range(chunk_size):
                    outs_test = outs_val = outs_val_left = outs_test_left = None

                    # banner por modelo (compacto)
                    if compact:
                        log_kv_block(
                            f"Model {idx+1}/{chunk_size}",
                            {"attempts": f"0..{max_retries}", "latent_mode": latent_mode, "snaps": ("yes" if want_snaps else "no")},
                            width=W,
                        )
                    else:
                        logger.info("training model=%d/%d", idx + 1, chunk_size)

                    for attempt in range(max_retries + 1):
                        try:
                            if not compact:
                                logger.info(
                                    "training model=%d/%d attempt=%d/%d",
                                    idx + 1, chunk_size, attempt + 1, max_retries + 1
                                )

                            model, _history = build_fn(
                                arch,
                                kind,
                                train_kwargs,
                                data_inputs,
                                epochs,
                                batch_size,
                                patience,
                                learning_rate,
                            )

                            # Main splits
                            outs_test = _predict_split(model, X_test, "test")
                            outs_val = _predict_split(model, X_val, "val")

                            # coupling (offline_analytic only, never in HPO)
                            if (not is_hpo) and latent_mode == "offline_analytic":
                                stage_banner("3.1", "Coupling", f"mode={latent_cfg.get('arps_coupling_mode','none')}", width=W)
                                # snapshot antes/depois (só sinais)
                                pre = {}
                                try:
                                    pre = {
                                        "val_q0": _safe_float(((outs_val or {}).get("anchor") or {}).get("q0_phys")),
                                        "test_q0": _safe_float(((outs_test or {}).get("anchor") or {}).get("q0_phys")),
                                        "val_b": _safe_float(((outs_val or {}).get("analytic_params") or {}).get("b")),
                                        "test_b": _safe_float(((outs_test or {}).get("analytic_params") or {}).get("b")),
                                    }
                                except Exception:
                                    pre = {}

                                try:
                                    outs_val, outs_test = _maybe_coupling_offline_analytic(
                                        outs_val=outs_val,
                                        outs_test=outs_test,
                                        scaler_y=scaler_target,
                                        split_recon_lengths=split_recon_lengths,
                                        latent_cfg=latent_cfg,
                                        logger=logger,
                                    )
                                except Exception as coup_ex:
                                    logger.exception(
                                        "arps_coupling_failed error=%s (continuing with uncoupled predictions)",
                                        str(coup_ex),
                                    )

                                post = {}
                                try:
                                    post = {
                                        "val_pred_shape": _shape((outs_val or {}).get("pred")),
                                        "test_pred_shape": _shape((outs_test or {}).get("pred")),
                                    }
                                except Exception:
                                    post = {}

                                # bloco compact: âncoras + b (alta utilidade pro bug)
                                kv = {}
                                kv.update({f"pre.{k}": v for k, v in (pre or {}).items()})
                                kv.update({f"post.{k}": v for k, v in (post or {}).items()})
                                log_kv_block("Coupling snapshot", kv, width=W)

                            # Visual-only fallback (disabled for offline_analytic)
                            _synthesize_val_members_if_missing(outs_val_=outs_val, outs_test_=outs_test)

                            # Capture spaghetti ASAP
                            _capture_members_once(outs_val_=outs_val, outs_test_=outs_test)

                            # Rank position summary (alto-sinal pro seu bug em VAL)
                            if latent_mode == "offline_analytic":
                                rs_val = _rank_summary_if_possible(split="val", outs=outs_val)
                                if rs_val:
                                    log_kv_block("Rank position — VAL (members vs pred)", rs_val, width=W)
                                rs_test = _rank_summary_if_possible(split="test", outs=outs_test)
                                if rs_test:
                                    log_kv_block("Rank position — TEST (members vs pred)", rs_test, width=W)

                            # Optional debug (first model only)
                            if idx == 0 and attempt == 0:
                                try:
                                    y_val_arr = data_inputs.get("y_val")
                                    if y_val_arr is not None and isinstance(outs_val, dict) and "pred" in outs_val:
                                        _log_val_boundary_debug(
                                            logger=logger,
                                            X_val=X_val,
                                            y_val=y_val_arr,
                                            pred_val=outs_val["pred"],
                                            scaler_target=scaler_target,
                                            target_idx_in_X=7,
                                            n_samples=3,
                                        )
                                except Exception as dbg_ex:
                                    logger.warning("[ValBoundaryDebug] failed with %s", str(dbg_ex))

                            # LEFT splits
                            if has_left_val and X_val_left is not None:
                                outs_val_left = _predict_split(model, X_val_left, "val_left")
                            if has_left_test and X_test_left is not None:
                                outs_test_left = _predict_split(model, X_test_left, "test_left")

                            break

                        except Exception as ex:
                            _cleanup_session(reason="train_or_predict_failed")
                            err(
                                "model=%d/%d attempt=%d/%d failed with %s: %s",
                                idx + 1, chunk_size, attempt + 1, max_retries + 1,
                                type(ex).__name__, str(ex),
                            )
                            logger.error(
                                "Traceback:\n%s",
                                "".join(traceback.format_exception(type(ex), ex, ex.__traceback__)),
                            )

                            if attempt == max_retries:
                                if skip_on_failure:
                                    warn("skip_on_failure model=%d/%d", idx + 1, chunk_size)
                                    outs_test = outs_val = None
                                    break
                                raise
                            else:
                                warn("retry reason=%s", type(ex).__name__)

                    if outs_test is None or outs_val is None:
                        continue

                    successful += 1

                    # Aggregate AFTER capture
                    _accumulate(outs_test, running_test, "test", successful)
                    _accumulate(outs_val, running_val, "val", successful)

                    if outs_val_left is not None:
                        _accumulate(outs_val_left, running_val, "val_left", successful)
                    if outs_test_left is not None:
                        _accumulate(outs_test_left, running_test, "test_left", successful)

                    # Alpha (optional)
                    if isinstance(outs_test, dict) and "alpha" in outs_test:
                        try:
                            running_alpha_sum += float(outs_test["alpha"])
                            models_with_alpha += 1
                        except Exception:
                            logger.debug("alpha_present_but_not_scalar; ignoring for mean")

                    _cleanup_session(reason="between_models")

                if successful == 0:
                    raise RuntimeError("No models succeeded in this chunk")

                out: Dict[str, Any] = {
                    "successful_models": successful,
                    "alpha": (running_alpha_sum / models_with_alpha) if models_with_alpha else None,
                }
                out.update(running_test)
                out.update(running_val)

                # Meta (non-breaking)
                out_meta = dict(out.get("meta") or {})
                out_meta.update({
                    "latent_cfg": latent_cfg,
                    "split_recon_lengths": split_recon_lengths,
                    "has_left_val": bool(has_left_val),
                    "has_left_test": bool(has_left_test),
                    "run_kind": run_kind,
                    "is_hpo": bool(is_hpo),
                })

                # Publish spaghetti members (SCALED)
                if captured_val_members_scaled is not None:
                    out["pred_members_val_scaled"] = captured_val_members_scaled
                    out["pred_members_val_meta"] = captured_val_members_meta or {}
                    out["pred_members_val"] = captured_val_members_scaled  # legacy alias (still scaled)
                    out_meta["integrated_view_val_members_scaled"] = captured_val_members_scaled
                    out_meta["integrated_view_val_members_meta"] = captured_val_members_meta or {}

                if captured_test_members_scaled is not None:
                    out["pred_members_scaled"] = captured_test_members_scaled
                    out["pred_members_meta"] = captured_test_members_meta or {}
                    out["pred_members"] = captured_test_members_scaled  # legacy alias (still scaled)
                    out_meta["integrated_view_test_members_scaled"] = captured_test_members_scaled
                    out_meta["integrated_view_test_members_meta"] = captured_test_members_meta or {}

                out["meta"] = out_meta

                stage_banner(4, "Chunk summary", "final recap", width=W)
                log_kv_block(
                    "Executor summary",
                    {
                        "successful_models": successful,
                        "alpha": (out.get("alpha") if out.get("alpha") is not None else "None"),
                        "snaps": ("yes" if want_snaps else "no"),
                        "left_val": ("yes" if has_left_val else "no"),
                        "left_test": ("yes" if has_left_test else "no"),
                        "spaghetti_val": ("yes" if captured_val_members_scaled is not None else "no"),
                        "spaghetti_test": ("yes" if captured_test_members_scaled is not None else "no"),
                        "latent_mode": latent_mode,
                    },
                    width=W,
                )

                ok("executor_done successful_models=%d latent_mode=%s", successful, latent_mode)
                return out

            finally:
                _cleanup_session(reason="finalize_executor")






