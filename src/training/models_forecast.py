# Standard library imports
import os

# Third-party imports
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.models import load_model

# Local imports
from models import create_model

import tensorflow as tf

# Configure threads for parallel execution
tf.config.threading.set_intra_op_parallelism_threads(24)  # For within operations
tf.config.threading.set_inter_op_parallelism_threads(24)  # For between operations


def warmup_and_decay_lr(epoch, lr, warmup_epochs=10, decay_start=20, initial_lr=0.001, min_lr=1e-5):
    """Função que implementa Warmup + Decay na taxa de aprendizado."""
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    elif epoch >= decay_start:
        return max(initial_lr * 0.1**((epoch - decay_start) / 15), min_lr)  # Ajuste o divisor para 30
    return lr



# ─────────────────────────────────────────────────────────────────────────────
# Train/fit utilities (checkpoint in‑memory)
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model,
    X_train,
    y_train,
    model_path,
    *,
    fine_tune: bool,
    batch_size_override: int | None = None,
):
    import tempfile

    # Early stopping common
    early_stopping = EarlyStopping(
        monitor="val_mean_absolute_error", patience=5 if fine_tune else 300, restore_best_weights=True, verbose=0
    )

    # Disable heavy disk I/O during fine‑tune
    if fine_tune:
        checkpoint = None
        ckpt_list = []
    else:
        tmp_ckpt = tempfile.NamedTemporaryFile(delete=False).name
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            tmp_ckpt, monitor="val_mean_absolute_error", save_best_only=True, verbose=0
        )
        ckpt_list = [checkpoint]

    bs = batch_size_override or (4 if fine_tune else 16)
    history = model.fit(
        X_train,
        y_train,
        epochs=15 if fine_tune else 300,
        batch_size=bs,
        validation_split=0.05,
        verbose=0,
        shuffle=False,
        callbacks=[c for c in [early_stopping, *ckpt_list] if c],
    )

    if not fine_tune and checkpoint is not None:
        model.load_weights(checkpoint.filepath)
        model.save(model_path)
    return model

def train_and_evaluate_disruptive(
    X_train_DL,
    y_train_DL,
    model_path: str,
    fine_tune: bool = True,
    well=None,
    architecture_name: str = "Generic",
    batch_size_override: int | None = None,
    model_creation_kwargs: dict = None
):
    """Robust train / fine‑tune that avoids temp‑file issues.

    * Accepts `batch_size_override (for base training).
    * Falls back to an in‑memory training routine if the internal
      `train_model helper raises an I/O error (e.g. temp path not a dir).
    """

    model_creation_kwargs = model_creation_kwargs or {}

    if fine_tune and os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = create_model(architecture_name=architecture_name, 
                             input_shape=X_train_DL,
                             **model_creation_kwargs)

    # First attempt: delegate to helper (uses checkpoint logic)
    try:
        model = train_model(
            model,
            X_train_DL,
            y_train_DL,
            model_path,
            fine_tune=fine_tune,
            batch_size_override=batch_size_override,
        )
    except Exception as err:
        # Fallback: pure in‑memory training (no disk checkpoint)
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        print("[WARN] train_model fallback due to:", err)
        bs = batch_size_override or (4 if fine_tune else 16)
        early = EarlyStopping("val_mean_absolute_error", patience=5 if fine_tune else 300, restore_best_weights=True, verbose=0)
        model.fit(
            X_train_DL,
            y_train_DL,
            epochs=15 if fine_tune else 300,
            batch_size=bs,
            validation_split=0.05,
            verbose=0,
            shuffle=False,
            callbacks=[early],
        )
        if not fine_tune:
            # Persist final model once at end of base training
            model.save(model_path)

    return model



def train_and_evaluate_XGB(X_train, y_train):
    """
    Trains a Gradient Boosting Regressor (GBR) model with random cross-validation,
    evaluates its performance on the test set, and visualizes the results.

    Args:
        X_train (NumPy array): Training features.
        y_train (NumPy array): Training target values.
        X_test (NumPy array): Test features.
        y_test (NumPy array): Test target values.

    Returns:
        None
    """
    
    # Define the parameter grid for random search
    param_grid = {
        'xgbregressor__n_estimators': [100, 200, 500],
        'xgbregressor__learning_rate': [0.001, 0.01, 0.05],
        'xgbregressor__max_depth': [4, 5, 6, 7],
        'xgbregressor__min_child_weight': [1, 3, 5],
        'xgbregressor__subsample': [0.8, 0.9, 1.0],
        'xgbregressor__gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'xgbregressor__reg_alpha': [0, 0.1, 0.5, 1, 1.5],
        'xgbregressor__reg_lambda': [0, 0.1, 0.5, 1, 1.5, 2]
    }

    # Create the pipeline
    pipeline = Pipeline([
        ('xgbregressor', XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=1, tree_method="hist"))
    ])
    
    # Initialize the RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        cv=3,
        n_iter=30,
        n_jobs=24,
        random_state=42
    )
    
    X_train = X_train[-50:]
    y_train = y_train[-50:]

    # Fit the random search model on the training data
    random_search.fit(X_train, y_train)
    
    # Get the best parameters and their corresponding score
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    # Print the best parameters and score
    # print(f"Best parameters: {best_params}")
    # print(f"Best cross-validation score: {best_score}")
    
    # Após o ajuste do modelo, obter o melhor modelo encontrado
    best_gb_regressor = random_search.best_estimator_

    return best_gb_regressor

# -*- coding: utf-8 -*-
"""Incremental XGB training (v2)
-------------------------------------------------
Mudanças nesta versão
=====================
1. **Cache** de Booster por poço (`_XGB_CACHE`).
2. **Crescimento controlado**: fatiamos para manter no máximo `MAX_TREES` árvores.
3. **Reset automático** quando ultrapassa `RESET_EVERY` árvores.
4. **Atualização adaptativa**: nº de novas árvores proporcional ao tamanho atual.
5. **Check‑point** em disco apenas a cada `SAVE_EVERY` iterações.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# ╭────────────────────────────────────────────────────────────────────────╮
#  Configurações globais                                                   │
# ╰────────────────────────────────────────────────────────────────────────╯
_XGB_CACHE: dict[str, xgb.Booster] = {}
SAVE_EVERY   = 100   # grava em disco
MAX_TREES    = 4000  # fatiar para este limite
RESET_EVERY  = 5000  # re‑treinar do zero se atingir

def train_and_evaluate_XGB(
    X_train: np.ndarray,
    y_train: np.ndarray,
    control_iteration: int,
    *,
    model_path: str,
    param_grid: Dict[str, Any] | None = None,
    n_iter: int = 30,
    cv: int = 3,
    update_rounds: int | None = None,  # calculado se None
    rng: int = 42,
):
    """Baseline + atualização incremental com controle de tamanho.

    ▸ **Iteração 0** (ou modelo inexistente) → RandomizedSearchCV.
    ▸ **Demais** → acrescenta árvores (warm‑start) com:
        • *update_rounds* adaptativo
        • *slice* para manter até `MAX_TREES`
        • *reset* completo caso chegue a `RESET_EVERY`
    """

    # ───────────────────── Preparação dos dados ──────────────────────
    dir_name = os.path.dirname(model_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    X_train, y_train = X_train[-50:], y_train[-50:]  # janela um pouco maior

    # ───────────────────── Fase 1 – baseline  ────────────────────────
    if control_iteration == 0 or not Path(model_path).exists():
        if param_grid is None:
            param_grid = {
            'xgbregressor__n_estimators': [100, 200, 500],
            'xgbregressor__learning_rate': [0.001, 0.01, 0.05],
            'xgbregressor__max_depth': [4, 5, 6, 7],
            'xgbregressor__min_child_weight': [1, 3, 5],
            'xgbregressor__subsample': [0.8, 0.9, 1.0],
            'xgbregressor__gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'xgbregressor__reg_alpha': [0, 0.1, 0.5, 1, 1.5],
            'xgbregressor__reg_lambda': [0, 0.1, 0.5, 1, 1.5, 2]
        }
        base_est = XGBRegressor(
            objective="reg:squarederror",
            random_state=rng,
            n_jobs=1,
            tree_method="hist",
            max_bin=64,
        )
        pipe = Pipeline([("xgbregressor", base_est)])
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            n_jobs=os.cpu_count() - 1,
            random_state=rng,
            verbose=1,
        )
        search.fit(X_train, y_train)

        best_est: XGBRegressor = search.best_estimator_.named_steps["xgbregressor"]
        booster = best_est.get_booster()
        booster.save_model(model_path)
        with open(model_path + ".params.json", "w", encoding="utf8") as fp:
            json.dump(best_est.get_params(), fp)

        _XGB_CACHE[model_path] = booster
        return best_est

    # ───────────────────── Fase 2 – incremental ──────────────────────
    booster = _XGB_CACHE.get(model_path)
    if booster is None:
        booster = xgb.Booster()
        booster.load_model(model_path)

    current_trees = booster.num_boosted_rounds()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    booster = xgb.train({}, dtrain, num_boost_round=update_rounds, xgb_model=booster)

    # ─── slice se exceder MAX_TREES ───────────────────────────────────
    if booster.num_boosted_rounds() > MAX_TREES:
        start = booster.num_boosted_rounds() - MAX_TREES
        booster = booster[start:]  # slice seguro (início ≥ 0)

    # ─── reset completo se exceder RESET_EVERY ────────────────────────
    if booster.num_boosted_rounds() > RESET_EVERY:
        print(f"[XGB] Resetting model (>{RESET_EVERY} trees).")
        booster = xgb.train({}, dtrain, num_boost_round=20)

    # checkpoint em disco
    if (control_iteration % SAVE_EVERY == 0):
        booster.save_model(model_path)

    _XGB_CACHE[model_path] = booster  # update cache

    # wrapper sklearn
    wrapped = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=booster.num_boosted_rounds(),
        tree_method="hist",
    )
    wrapped._Booster = booster
    wrapped._le = None
    return wrapped





