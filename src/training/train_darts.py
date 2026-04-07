#src/training/train_darts.py
from __future__ import annotations

import inspect
import logging
import os
import random
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import torch
from darts import TimeSeries
from darts.models import (
    # Deep learning models
    NHiTSModel,
    TiDEModel,
    NLinearModel,
    NBEATSModel,
    # Classical models
    ARIMA,
    AutoARIMA,
    LinearRegressionModel,
)

# === Lightweight TXT logger (JSON lines) ======================================
# Set DARTS_DEBUG_TXT env var to change the output location (default: ./darts_debug.txt)
import json, datetime
DEBUG_TXT = os.environ.get("DARTS_DEBUG_TXT", "./darts_debug.txt")

def _dbg__type(x):
    try:
        return type(x).__name__
    except Exception:
        return "<unknown>"

# ==============================================================================
# Utilities
# ==============================================================================

def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set global RNG seeds for Python, NumPy, and PyTorch (CPU/CUDA) to promote reproducible runs.

    This also toggles PyTorch's deterministic mode (when `deterministic=True`) and disables cuDNN
    benchmarking to reduce nondeterminism in convolutional kernels.

    Args:
        seed: Integer seed to apply across Python's `random`, NumPy, and PyTorch RNGs.
        deterministic: If True, enable PyTorch deterministic algorithms (with `warn_only=True`)
            and set cuDNN to deterministic mode with benchmarking disabled.

    Notes:
        • Some GPU ops are still non-deterministic unless you additionally set
          `CUBLAS_WORKSPACE_CONFIG` as described by PyTorch/CuBLAS docs.
        • Deterministic execution can slow down training.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _is_deep(model_key: str) -> bool:
    """
    Return True if `model_key` corresponds to a Darts deep learning model.

    Deep models (e.g., TiDE, N-HiTS, NLinear, N-Beats) use PyTorch Lightning under
    the hood and accept training knobs such as `n_epochs`, `pl_trainer_kwargs`, etc.

    Args:
        model_key: Model name key as used in MODEL_REGISTRY (e.g., "TiDE", "NHiTS").

    Returns:
        bool: True for deep models, False for classical/statistical ones.
    """
    return model_key in {"NHiTS", "TiDE", "TiDE_RIN", "N-Beats", "NLinear"}


def _split_architecture_name(architecture_name: str) -> Tuple[str, str]:
    """
    Split an architecture label like 'Darts_TiDE' into ('Darts', 'TiDE').

    Args:
        architecture_name: A string with the pattern '<library>_<model>',
            e.g. "Darts_TiDE" or "Darts_NLinear".

    Returns:
        Tuple[str, str]: (library, model) pair.

    Raises:
        ValueError: If the string does not contain an underscore.
    """
    if "_" not in architecture_name:
        raise ValueError(f"architecture_name must look like 'Darts_TiDE', got {architecture_name}")
    lib, model = architecture_name.split("_", 1)
    return lib, model


def _signature_info(model_cls) -> Tuple[set, bool]:
    """
    Inspect a model class constructor to learn its explicit parameters and whether it accepts **kwargs.

    Args:
        model_cls: The Darts model class (not instance), e.g., `TiDEModel`.

    Returns:
        Tuple[set, bool]:
            - A set with all explicit parameter names from `__init__`.
            - A boolean indicating if `__init__` has a VAR_KEYWORD (**kwargs) parameter.

    Notes:
        We use this to safely filter which keys from our config can be forwarded
        into each model constructor without raising unexpected-argument errors.
    """
    sig = inspect.signature(model_cls.__init__)
    names = set(sig.parameters.keys())
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    return names, has_var_kw


# training/train_darts.py
from pytorch_lightning.callbacks import EarlyStopping

def _coerce_trainer_kwargs(*, accelerator: str, n_epochs: int, user_kwargs: Optional[Dict[str, Any]] = None,
                           patience: Optional[int] = None) -> Dict[str, Any]:
    kw = dict(user_kwargs or {})
    kw["deterministic"] = True
    kw["max_epochs"] = int(n_epochs)

    acc = str(accelerator).lower()
    if acc in {"cpu", "none"}:
        kw["accelerator"] = "cpu"; kw["devices"] = 1; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    elif acc in {"gpu", "cuda"}:
        kw["accelerator"] = "gpu"; kw.setdefault("devices", 1)
    else:
        kw.setdefault("accelerator", "auto")

    # inject patience if user didn’t add callbacks
    if patience and "callbacks" not in kw:
        kw["callbacks"] = [EarlyStopping(monitor="val_loss", mode="min", patience=int(patience))]
    kw.setdefault("enable_progress_bar", False)
    return kw



def _kv_table(rows: List[Tuple[str, Any]]) -> str:
    """
    Render a tiny ASCII table from (key, value) rows for readable logging.

    Args:
        rows: Iterable of pairs like `[("n_epochs", 10), ("accelerator", "cpu")]`.

    Returns:
        str: A formatted table string, or "(empty)" if no rows were provided.

    Example:
        >>> print(_kv_table([("a", 1), ("b", 2)]))
        +---+---+
        | key | value |
        +---+---+
        | a   | 1     |
        | b   | 2     |
        +---+---+
    """
    if not rows:
        return "(empty)"
    k_w = max(len(str(k)) for k, _ in rows)
    v_w = max(len(str(v)) for _, v in rows)
    line = "+-" + "-" * k_w + "-+-" + "-" * v_w + "-+"
    out = [line, f"| {'key'.ljust(k_w)} | {'value'.ljust(v_w)} |", line]
    for k, v in rows:
        out.append(f"| {str(k).ljust(k_w)} | {str(v).ljust(v_w)} |")
    out.append(line)
    return "\n".join(out)


# ==============================================================================
# Model registry
# ==============================================================================

# ----------------------------- model registry -------------------------------- #
MODEL_REGISTRY = {
    "NHiTS": NHiTSModel,
    "TiDE": TiDEModel,
    "TiDE_RIN": TiDEModel,  # alias
    "N-Beats": NBEATSModel,
    "NLinear": NLinearModel,
    "ARIMA": ARIMA,
    "AutoARIMA": AutoARIMA,
    "LinearRegression": LinearRegressionModel,
}

# Which models support past_covariates according to the docs:
PAST_COVARIATES_SUPPORT = {
    "TiDE": True,
    "TiDE_RIN": True,
    "NHiTS": True,
    "NLinear": False,
    "LinearRegression": True,   # requires proper lags_past_covariates in ctor
    "ARIMA": False,
    "AutoARIMA": False,
    "N-Beats": True,
}

def _model_supports_past_covariates(model_key: str) -> bool:
    return bool(PAST_COVARIATES_SUPPORT.get(model_key, False))



# ==============================================================================
# Construct & train
# ==============================================================================

# Pipeline-only keys that must never be forwarded to Darts constructors
_PIPELINE_ONLY_KEYS = {
    "test_size", "val_size", "seed", "architecture_name", "profile",
    "selected_features", "target_column", "use_past_covariates",
}

def _build_ctor_kwargs_for_model(
    model_cls,
    cfg: Dict[str, Any],
    *,
    is_deep: bool,
    input_chunk_length: int,
    output_chunk_length: int,
    n_epochs_eff: Optional[int],
    learning_rate: Optional[float],
    batch_size: Optional[int],
    seed: Optional[int],
    pl_trainer_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    explicit_params, has_var_kw = _signature_info(model_cls)
    ctor_kwargs: Dict[str, Any] = {}

    """
    Build a safe keyword-argument dictionary for a Darts model constructor.

    This function inspects the model's `__init__` signature and forwards only
    compatible keys, guarding against "unexpected keyword argument" errors.
    For deep models (those with **kwargs), we also pass training-time knobs (epochs,
    optimizer lr, batch size, trainer kwargs) when supported.

    Args:
        model_cls: The Darts model class (e.g., `TiDEModel`).
        cfg: The full run configuration (pipeline + model hyperparameters).
        is_deep: Whether this is a deep learning model (uses PL under the hood).
        input_chunk_length: Desired input window length.
        output_chunk_length: Desired forecast horizon length.
        n_epochs_eff: Effective number of epochs for deep models (None for classical).
        learning_rate: Learning rate to inject into `optimizer_kwargs` (deep models).
        batch_size: Batch size to forward if the constructor accepts it.
        seed: Seed to map to `random_state` if accepted.
        pl_trainer_kwargs: Lightning `Trainer` kwargs (epochs, accelerator, devices).

    Returns:
        Dict[str, Any]: Keyword args safe to pass into the model constructor.

    Example:
        >>> ctor = _build_ctor_kwargs_for_model(
        ...     TiDEModel, cfg={"hidden_size": 64, "n_epochs": 10},
        ...     is_deep=True, input_chunk_length=100, output_chunk_length=100,
        ...     n_epochs_eff=10, learning_rate=1e-3, batch_size=32, seed=42,
        ...     pl_trainer_kwargs={"accelerator": "cpu", "max_epochs": 10}
        ... )
        >>> model = TiDEModel(**ctor)
    """

    # Core chunk lengths only if ctor accepts them
    if "input_chunk_length" in explicit_params:
        ctor_kwargs["input_chunk_length"] = input_chunk_length
    if "output_chunk_length" in explicit_params:
        ctor_kwargs["output_chunk_length"] = output_chunk_length

    # Model-specific params from cfg (ONLY explicit ctor args)
    for k, v in cfg.items():
        if k in _PIPELINE_ONLY_KEYS:
            continue
        if k in explicit_params:
            ctor_kwargs[k] = v

    # Deep-only knobs (if accepted or **kwargs exists)
    if is_deep:
        if (n_epochs_eff is not None) and ("n_epochs" in explicit_params or has_var_kw):
            ctor_kwargs["n_epochs"] = int(n_epochs_eff)
        if (batch_size is not None) and ("batch_size" in explicit_params or has_var_kw):
            ctor_kwargs["batch_size"] = int(batch_size)
        if (seed is not None) and ("random_state" in explicit_params or has_var_kw):
            ctor_kwargs["random_state"] = int(seed)

        opt_kw = dict(cfg.get("optimizer_kwargs", {}))
        if learning_rate is not None:
            opt_kw.setdefault("lr", float(learning_rate))
        if opt_kw and ("optimizer_kwargs" in explicit_params or has_var_kw):
            ctor_kwargs["optimizer_kwargs"] = opt_kw

        if pl_trainer_kwargs and ("pl_trainer_kwargs" in explicit_params or has_var_kw):
            ctor_kwargs["pl_trainer_kwargs"] = pl_trainer_kwargs

        # harmless convenience flags when ctor can swallow them
        if "save_checkpoints" in explicit_params or has_var_kw:
            ctor_kwargs.setdefault("save_checkpoints", False)
        if "force_reset" in explicit_params or has_var_kw:
            ctor_kwargs.setdefault("force_reset", True)
        if "work_dir" in explicit_params or has_var_kw:
            ctor_kwargs.setdefault("work_dir", "/tmp/darts_noop")

    return ctor_kwargs

def _as_int(x, fallback: int | None = None) -> int:
    try:
        return int(x)
    except Exception:
        if fallback is None:
            raise
        return int(fallback)

def _as_int_list(x) -> list[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in x]
    return [int(x)]

def _sanitize_nhits(cfg: Dict[str, Any]) -> None:
    # num_stacks/blocks/layers must be ints
    for k in ("num_stacks", "num_blocks", "num_layers"):
        if k in cfg:
            cfg[k] = _as_int(cfg[k])

    # layer_widths: int or list[int] with len == num_stacks
    if "layer_widths" in cfg:
        lw = cfg["layer_widths"]
        ns = _as_int(cfg.get("num_stacks", 1))
        if isinstance(lw, (list, tuple, np.ndarray)):
            lw = [int(v) for v in lw]
            if len(lw) == 1 and ns > 1:
                lw = lw * ns
        else:
            lw = _as_int(lw)
            if ns > 1:
                lw = [lw] * ns
        cfg["layer_widths"] = lw

def _sanitize_tide(cfg: Dict[str, Any], output_chunk_length: int) -> None:
    INT_KEYS = (
        "hidden_size", "temporal_decoder_hidden_size",
        "num_encoder_layers", "num_decoder_layers",
        "temporal_width_past", "temporal_width_future",
        "n_heads", "attention_heads",  # caso apareça em perfis
    )
    for k in INT_KEYS:
        if k in cfg:
            try:
                cfg[k] = int(cfg[k])
            except Exception:
                # fallback seguro
                cfg[k] = 1 if ("layer" in k or "width" in k or "head" in k) else output_chunk_length

    # mínimos
    for k in ("num_encoder_layers", "num_decoder_layers", "temporal_width_past", "temporal_width_future"):
        if k in cfg:
            cfg[k] = max(1, int(cfg[k]))

    # decoder_output_dim: int > 0; se vier lista/tupla singleton, desembrulhe
    if "decoder_output_dim" in cfg:
        val = cfg["decoder_output_dim"]
        if isinstance(val, (list, tuple)) and len(val) == 1:
            val = val[0]
        try:
            val = int(val)
            if val <= 0:
                raise ValueError
        except Exception:
            val = int(output_chunk_length)
        cfg["decoder_output_dim"] = val

def _sanitize_lr(cfg: Dict[str, Any], input_chunk_length: int, output_chunk_length: int, using_past_covs: bool) -> None:
    # target lags: coerce to int even if provided
    if "lags" in cfg:
        v = cfg["lags"]
        cfg["lags"] = _as_int(v) if not isinstance(v, (list, tuple, np.ndarray)) else [int(x) for x in v]
    else:
        cfg["lags"] = int(input_chunk_length)

    # past covariates lags
    if using_past_covs:
        if "lags_past_covariates" in cfg:
            cfg["lags_past_covariates"] = _as_int_list(cfg["lags_past_covariates"])
        else:
            cfg["lags_past_covariates"] = list(range(-int(input_chunk_length), 0))
    else:
        cfg["lags_past_covariates"] = []

    # future covariates off -> force []
    if cfg.get("lags_future_covariates", []) is None:
        cfg["lags_future_covariates"] = []

    # multi-step regression stability
    cfg["multi_models"] = True
    cfg["output_chunk_length"] = int(output_chunk_length)

def _sanitize_arima(cfg: Dict[str, Any]) -> None:
    # Coerce (p,d,q)
    if "p" in cfg or "d" in cfg or "q" in cfg:
        for k in ("p", "d", "q"):
            if k in cfg:
                cfg[k] = _as_int(cfg[k], fallback=0)

    # Seasonal handling
    # If `seasonal_order` exists but is not a 4-tuple/list, drop it (let Darts default).
    so = cfg.get("seasonal_order", None)
    if so is not None:
        if not (isinstance(so, (list, tuple)) and len(so) == 4):
            cfg.pop("seasonal_order", None)

    # If user provided m (season length) but no valid seasonal_order, add a safe default
    if "m" in cfg:
        m = _as_int(cfg["m"])
        if m <= 0:
            cfg["m"] = 0
        if "seasonal_order" not in cfg and m > 1:
            # conservative seasonal spec
            cfg["seasonal_order"] = (1, 0, 0, m)


import inspect

def _filter_kwargs_by_signature(func, kwargs: dict) -> dict:
    """
    Remove kwargs que não existem na assinatura de `func`.
    Funciona com métodos bound (ex.: model.fit).
    """
    try:
        sig = inspect.signature(func)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        # fallback conservador: não filtra
        return kwargs


def train_darts_model(
    train_series: TimeSeries,
    val_series: TimeSeries,
    model_type: str,
    output_chunk_length: int,
    hyperparam_config: Dict[str, Any],
    train_covariates: TimeSeries | None = None,
    val_covariates: TimeSeries | None = None,
    *,
    use_past_covariates: bool,
) -> object:
    """
    Construct and fit a Darts model (deep or classical) with clean separation of concerns.

    IMPORTANT (Keras-equivalence note):
      • `val_series` may include EXACT `input_chunk_length` steps of *train* prepended as INPUT context.
      • This does NOT leak targets: the first validation target begins at the first true validation step.
      • If `val_series` is None (e.g., len(val) < horizon), the model trains without validation,
        mirroring Keras when there are 0 validation windows.
    """
    cfg = dict(hyperparam_config)  # do not mutate caller
    is_deep = _is_deep(model_type)

    accelerator    = cfg.get("accelerator", "cpu")
    batch_size     = int(cfg.get("batch_size", 16))
    patience       = int(cfg.get("patience", 50))
    learning_rate  = float(cfg.get("learning_rate", 1e-3))
    seed           = int(cfg.get("seed", 42))
    input_len      = int(cfg.get("input_chunk_length", output_chunk_length))

    # Deep-only trainer knobs
    n_epochs_eff: Optional[int] = None
    pl_trainer_kwargs: Optional[Dict[str, Any]] = None
    if is_deep:
        n_epochs_eff = int(cfg.get("n_epochs", cfg.get("max_epochs", 100)))
        pl_trainer_kwargs = _coerce_trainer_kwargs(
            accelerator=accelerator,
            n_epochs=n_epochs_eff,
            user_kwargs=cfg.get("pl_trainer_kwargs", {}),
        )

    ModelClass = MODEL_REGISTRY[model_type]

    ctor_kwargs = _build_ctor_kwargs_for_model(
        ModelClass,
        cfg,
        is_deep=is_deep,
        input_chunk_length=input_len,
        output_chunk_length=output_chunk_length,
        n_epochs_eff=n_epochs_eff,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )



    # ---- LOG: plan
    plan_rows = [
        ("model", model_type),
        ("is_deep", is_deep),
        ("use_past_covariates", use_past_covariates),
        ("accelerator", (pl_trainer_kwargs or {}).get("accelerator", "n/a") if is_deep else "n/a"),
        ("devices", (pl_trainer_kwargs or {}).get("devices", "n/a") if is_deep else "n/a"),
        ("n_epochs", n_epochs_eff if is_deep else "n/a"),
        ("batch_size", batch_size if is_deep else "n/a"),
        ("learning_rate", learning_rate if is_deep else "n/a"),
        ("patience", patience if is_deep else "n/a"),
        ("input_chunk_length", input_len if "input_chunk_length" in ctor_kwargs else "n/a"),
        ("output_chunk_length", output_chunk_length if "output_chunk_length" in ctor_kwargs else "n/a"),
        ("seed", seed),
    ]
    logging.info("Resolved training plan:\n" + _kv_table(plan_rows))

    # TiDE safety: decoder_output_dim must be an INT (nn.Linear out_features).
    if model_type == "TiDE" and "decoder_output_dim" in ctor_kwargs:
        orig = ctor_kwargs["decoder_output_dim"]
        val = orig
        # unwrap singleton list/tuple like (100,) -> 100
        if isinstance(val, (list, tuple)) and len(val) == 1:
            val = val[0]
        # try to coerce to int; if it fails, fall back to horizon
        try:
            val = int(val)
        except Exception:
            val = None
        if not isinstance(val, int) or val <= 0:
            # safest: tie it to output_chunk_length
            val = output_chunk_length
        ctor_kwargs["decoder_output_dim"] = val


    model = ModelClass(**ctor_kwargs)

    if is_deep:
        dataloader_kwargs = {"num_workers": 1, "pin_memory": False, "drop_last": False}
        try:
            fit_kwargs = dict(
                series=train_series,
                val_series=val_series,
                past_covariates=(train_covariates if use_past_covariates else None),
                val_past_covariates=(val_covariates if use_past_covariates else None),
                verbose=False,
                dataloader_kwargs=dataloader_kwargs,
            )
            # --- NOVO: filtrar pelos args reais de fit() ---
            fit_kwargs = _filter_kwargs_by_signature(model.fit, fit_kwargs)
    
            model.fit(**fit_kwargs)
        except Exception as e:
            _dbg("fit_error", model_type=model_type, error=str(e))
            raise
    else:
        try:
            fit_kwargs = dict(
                series=train_series,
                past_covariates=(train_covariates if use_past_covariates else None),
                verbose=False,
            )
            # --- NOVO: idem aqui (ARIMA/AutoARIMA não aceitam past_covariates) ---
            fit_kwargs = _filter_kwargs_by_signature(model.fit, fit_kwargs)
    
            model.fit(**fit_kwargs)
        except Exception as e:
            _dbg("fit_error", model_type=model_type, error=str(e))
            raise


    

    return model


# ==============================================================================
# Public API used by the pipeline
# ==============================================================================

def _clamp_tide_widths(params: Dict[str, Any], n_covs: int) -> None:
    """
    Adjust TiDE's temporal widths to avoid feature-expansion warnings when covariates are few.

    If there are very few `past_covariates`, setting `temporal_width_past/future`
    higher than the number of covariate features triggers an internal expansion warning.
    This clamps widths to at most `n_covs - 1` (but ≥1) whenever possible.

    Args:
        params: Mutable params dict; keys `temporal_width_past` and `temporal_width_future`
            are updated in place if needed.
        n_covs: Number of covariate components available.

    Example:
        With 3 covariates and `temporal_width_past=4`, the function clamps it to 2.
    """
    if n_covs <= 0:
        return
    # prefer strictly less than n_covs to avoid expansion message
    safe_max = max(1, n_covs - 1)

    twp = int(params.get("temporal_width_past", 4))
    twf = int(params.get("temporal_width_future", 4))
    params["temporal_width_past"]   = max(1, min(twp, safe_max))
    params["temporal_width_future"] = max(1, min(twf, safe_max))


def _decide_use_past(model_key: str, params: Dict[str, Any], n_covs: int) -> bool:
    """
    Decide if we will use past_covariates for THIS run:
    - respect explicit params['use_past_covariates'] when present;
    - require model support and n_covs > 0.
    - for LinearRegression: if True and no lags_past_covariates provided, set a sane default.
    """
    supports = _model_supports_past_covariates(model_key)
    want = params.get("use_past_covariates", True if supports else False)
    will = bool(supports and want and (n_covs > 0))
    return will

# --- BEGIN PATCH: RegressionModel lag defaults & multi-step safety ---
def _ensure_lr_lags(params: dict, input_chunk_length: int, output_chunk_length: int, using_past_covs: bool) -> dict:
    """
    Normalize LR lag-related kwargs to avoid None and make multi-step consistent.
    - lags: int is fine; we default to input_chunk_length (interpreted as [-k, -1]).
    - lags_past_covariates: keep as-is if provided; else set to [] when not using past covs.
    - lags_future_covariates: must be [] (not None) when not using future covs to dodge HF edge-cases.
    - multi_models: force True for stability with multi-step outputs.
    """
    p = dict(params) if params else {}

    # Target lags
    if "lags" not in p or p["lags"] in (None, 0):
        p["lags"] = int(input_chunk_length)

    # Past covariates lags
    if using_past_covs:
        # Respect existing; otherwise mirror target window by default
        p.setdefault("lags_past_covariates", list(range(-input_chunk_length, 0)))
    else:
        p["lags_past_covariates"] = []

    # Future covariates not used -> MUST be []
    # (leaving it as None can trip the optimized HF index math)
    if "lags_future_covariates" in p and p["lags_future_covariates"] is None:
        p["lags_future_covariates"] = []

    # Multi-step regression stability
    # (single multi-output model can trigger corner-cases in optimized HF)
    p["multi_models"] = True

    # Be explicit on output length (you already are)
    p["output_chunk_length"] = int(output_chunk_length)

    return p

def _stack_ribbons(ts_list: List[TimeSeries]) -> np.ndarray:
    """
    Stack a list of Darts `TimeSeries` ribbons (each of length H) into a 2D NumPy array.

    Darts `historical_forecasts(last_points_only=False)` returns a list where each element
    is a short `TimeSeries` containing a horizon of predictions. This utility converts that
    list into a single array of shape `(n_samples, horizon)`.

    Args:
        ts_list: List of TimeSeries ribbons, each representing a forecast horizon.

    Returns:
        np.ndarray: Array of shape (n_samples, horizon). If Darts returns a degenerate
        shape (1D), we reshape it to 2D for consistency.

    Example:
        If you have 1263 ribbons each predicting 100 steps ahead:
            >>> arr = _stack_ribbons(pred_series_list)
            >>> arr.shape
            (1263, 100)
    """
    arr = np.array([ts.values() for ts in ts_list]).squeeze()
    if arr.ndim == 1:
        arr = arr.reshape(-1, len(arr))
    return arr


import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np

# --- Helpers to avoid Darts append() contiguity constraints -------------------
from typing import List
import pandas as pd

def _ts_from_values_like(ref: TimeSeries, values: np.ndarray) -> TimeSeries:
    """
    Build a new TimeSeries from raw values with a simple integer time axis (0..N-1),
    keeping the same component names as `ref`. This bypasses timestamp contiguity.
    """
    comps = list(getattr(ref, "components", []))
    times = pd.RangeIndex(start=0, stop=len(values), step=1)
    # values shape: (T,) or (T, K). Darts accepts both for univariate/multivariate.
    return TimeSeries.from_times_and_values(times=times, values=values, columns=comps or None)

def _concat_as_new_series(series_list: List[TimeSeries]) -> TimeSeries:
    """
    Concatenate multiple TimeSeries by values only (ignore original timestamps),
    producing a fresh series with a simple contiguous integer time axis.
    """
    if not series_list:
        raise ValueError("Empty series_list in _concat_as_new_series()")
    vals = np.concatenate([s.values(copy=False) for s in series_list], axis=0)
    return _ts_from_values_like(series_list[0], vals)

def _with_exact_context_series(
    *,
    target_series: TimeSeries,
    cov_series: Optional[TimeSeries],
    context_target_src: TimeSeries,
    context_cov_src: Optional[TimeSeries],
    k: int,
) -> Tuple[TimeSeries, Optional[TimeSeries]]:
    """
    Build a fresh validation target (and optional covariates) that PREPEND exactly k steps
    of train INPUT context, without using TimeSeries.append().
    """
    if k <= 0:
        tgt = _ts_from_values_like(target_series, target_series.values(copy=False))
        cov = (_ts_from_values_like(cov_series, cov_series.values(copy=False))
               if cov_series is not None else None)
        return tgt, cov

    tgt_vals = np.concatenate([context_target_src[-k:].values(copy=False),
                               target_series.values(copy=False)], axis=0)
    tgt = _ts_from_values_like(target_series, tgt_vals)

    if cov_series is not None and context_cov_src is not None:
        cov_vals = np.concatenate([context_cov_src[-k:].values(copy=False),
                                   cov_series.values(copy=False)], axis=0)
        cov = _ts_from_values_like(cov_series, cov_vals)
    else:
        cov = None

    return tgt, cov

# --- helper local, coloque no topo do arquivo (ou logo acima da função) ---
def _coerce_intlike_params(d: Dict[str, Any], keys: list[str]) -> list[str]:
    """
    Converte em int qualquer valor que seja numericamente inteiro:
    - '100' -> 100
    - 100.0 / np.float64(100) -> 100
    - bools são ignorados
    Retorna a lista de chaves efetivamente convertidas (para debug).
    """
    changed = []
    for k in keys:
        if k not in d:
            continue
        v = d[k]
        if isinstance(v, bool) or v is None:
            continue
        # ignore listas/tuplas (ex.: layer_widths) – essas têm tratamento próprio
        if isinstance(v, (list, tuple, dict)):
            continue
        try:
            fv = float(v)
            if fv.is_integer():
                iv = int(fv)
                if iv != v:
                    d[k] = iv
                    changed.append(k)
        except Exception:
            # tenta inteiro direto p/ strings tipo "100"
            try:
                iv = int(v)
                if iv != v:
                    d[k] = iv
                    changed.append(k)
            except Exception:
                pass
    return changed


def main_train_darts_model(
    architecture_name: str,
    train_kwargs: Dict[str, Any],
    data_inputs: Dict[str, Any],
    epochs: Optional[int],
    batch_size: Optional[int],
    patience: Optional[int],
    learning_rate: Optional[float],
) -> Tuple[Any, Dict, np.ndarray, np.ndarray]:
    """
    Train a Darts model and produce validation/test forecast ribbons via `historical_forecasts`,
    with Keras-equivalent sliding-window behavior (stride=1) and *no leakage*.

    Equivalence guarantees:
      • Validation during fit sees EXACTLY `input_chunk_length` steps of context from the end of train,
        so the first validation prediction starts at the first val step.
      • Validation requires only len(val) >= horizon (not input_len + horizon).
      • If len(val) < horizon, we disable validation in fit (equivalent to 0 val windows in Keras).
      • historical_forecasts for VAL/TEST start at the exact first indices of each split, stride=1.
    """
    # ---------------------------------------------------------------------------------------------
    # 0) Unpack series (accept both the pipeline's "X_*" keys and native "ts_*" Darts keys)
    # ---------------------------------------------------------------------------------------------
    if "X_train" in train_kwargs:
        logging.debug("Adapting pipeline data format (X_train/X_val) to Darts TimeSeries.")
        ts_train = train_kwargs.get("X_train")
        ts_val   = train_kwargs.get("X_val")
        ts_test  = data_inputs.get("ts_test")
    else:
        logging.debug("Using direct Darts format (ts_train/ts_val).")
        ts_train = train_kwargs.get("ts_train")
        ts_val   = train_kwargs.get("ts_val")
        ts_test  = data_inputs.get("ts_test")



    main_col: str = train_kwargs.get("main_col")
    params: Dict[str, Any] = dict(train_kwargs.get("params", {}))

    # ---------------------------------------------------------------------------------------------
    # 1) Runtime overrides (Jupyter, CLI, etc.)
    # ---------------------------------------------------------------------------------------------
    if epochs is not None:        params["n_epochs"] = int(epochs)
    if batch_size is not None:    params["batch_size"] = int(batch_size)
    if patience is not None:      params["patience"] = int(patience)
    if learning_rate is not None: params["learning_rate"] = float(learning_rate)

    # 1) Aliases legacy → nomes do Darts, se faltarem
    if "input_chunk_length" not in params and "lag_window" in params:
        params["input_chunk_length"] = params["lag_window"]
    if "output_chunk_length" not in params and "horizon" in params:
        params["output_chunk_length"] = params["horizon"]
    
    # 2) Coagir ints (inclui input/output, épocas, batch, ARIMA etc.)
    int_like_common = [
        "input_chunk_length", "output_chunk_length",
        "n_epochs", "epochs", "batch_size", "patience",
        "lags", "m",
        "p", "d", "q", "max_p", "max_q", "max_P", "max_Q",
        "num_stacks", "num_blocks", "num_layers",
        "ensemble_models",
    ]
    int_like_deep = [
        "hidden_size", "num_encoder_layers", "num_decoder_layers", "decoder_output_dim",
    ]
    _changed = _coerce_intlike_params(
        params,
        int_like_common + (int_like_deep if _is_deep(_split_architecture_name(architecture_name)[1]) else [])
    )


    logging.info(f"params: {params}")

    # ---------------------------------------------------------------------------------------------
    # 2) Split target + covariates
    # ---------------------------------------------------------------------------------------------
    ts_target_train, ts_covs_train = ts_train[main_col], ts_train.drop_columns(main_col)
    ts_target_val,   ts_covs_val   = ts_val[main_col],   ts_val.drop_columns(main_col)
    ts_target_test,  ts_covs_test  = ts_test[main_col],  ts_test.drop_columns(main_col)

    # ---------------------------------------------------------------------------------------------
    # 3) Seed and model key
    # ---------------------------------------------------------------------------------------------
    set_global_seed(int(params.get("seed", 42)), deterministic=True)
    lib, model_key = _split_architecture_name(architecture_name)

    

    if lib != "Darts":
        raise ValueError(f"Unsupported library '{lib}'. This trainer handles only Darts_* models.")
    is_deep = _is_deep(model_key)

    # ---------------------------------------------------------------------------------------------
    # 4) Covariate plan + model-specific tweaks
    # ---------------------------------------------------------------------------------------------
    n_covs = int(ts_covs_train.n_components) if ts_covs_train is not None else 0
    if is_deep and model_key in {"TiDE", "TiDE_RIN"}:
        _clamp_tide_widths(params, n_covs)

        # Ensure these are ints after clamping (in case profiles fed strings/tuples)
        for k in ("temporal_width_past", "temporal_width_future"):
            if k in params:
                v = params[k]
                if isinstance(v, (list, tuple)) and len(v) == 1:
                    v = v[0]
                try:
                    params[k] = int(v)
                except Exception:
                    # fallback: minimal safe value
                    params[k] = 1

    input_len = int(params.get("input_chunk_length", params.get("output_chunk_length")))
    horizon   = int(params.get("output_chunk_length"))
    use_past  = _decide_use_past(model_key, params, n_covs)

        # --- NEW: per-model param sanitation before training ---
    if model_key == "NHiTS":
        _sanitize_nhits(params)
    elif model_key in {"TiDE", "TiDE_RIN"}:
        _sanitize_tide(params, output_chunk_length=horizon)
    elif model_key == "LinearRegression":
        _sanitize_lr(params, input_chunk_length=input_len, output_chunk_length=horizon, using_past_covs=use_past)
    elif model_key in {"ARIMA", "AutoARIMA"}:
        _sanitize_arima(params)


    # LinearRegression lags if using past covariates
    if model_key == "LinearRegression" and use_past:
        # _ensure_lr_lags(params, input_len)
        params = _ensure_lr_lags(
            params=params,  # your current dict (or {})
            input_chunk_length=input_len,
            output_chunk_length=horizon,
            using_past_covs=True,
        )

    # ---------------------------------------------------------------------------------------------
    # 5) Build VALIDATION WITH EXACT CONTEXT (Keras-equivalent, NO LEAKAGE)
    #    - Prepend EXACTLY `input_len` steps of train to val for INPUT context.
    #    - Targets measured only inside the true val region.
    #    - If len(val) < horizon: disable validation (equivalent to 0 val windows in Keras).
    # ---------------------------------------------------------------------------------------------
    len_val = len(ts_target_val)
    if len_val >= horizon:
        val_target_for_fit, val_cov_for_fit = _with_exact_context_series(
            target_series=ts_target_val,
            cov_series=(ts_covs_val if use_past else None),
            context_target_src=ts_target_train,
            context_cov_src=(ts_covs_train if use_past else None),
            k=input_len,
        )
        logging.info(
            "Validation prepared with exact context (no-append path): len(val)=%d, input_len=%d, horizon=%d → OK",
            len_val, input_len, horizon
        )
    else:
        val_target_for_fit = None
        val_cov_for_fit = None
        logging.warning(
            "Validation disabled for fit(): len(val)=%d < horizon=%d. "
            "This mirrors Keras behavior where val would have 0 windows.",
            len_val, horizon
        )



    
    # ---------------------------------------------------------------------------------------------
    # 6) Train (fit)
    # ---------------------------------------------------------------------------------------------
    model = train_darts_model(
        train_series=ts_target_train,
        val_series=val_target_for_fit,                 # may include exact context, or None
        model_type=model_key,
        output_chunk_length=horizon,
        hyperparam_config=params,
        train_covariates=(ts_covs_train if use_past else None),
        val_covariates=(val_cov_for_fit if use_past else None),
        use_past_covariates=use_past,
    )

    logging.info("--- Generating historical forecasts (val/test) ---")

    # ---------------------------------------------------------------------------------------------
    # 7) historical_forecasts for VAL/TEST (stride=1), starting at the true split boundaries
    # ---------------------------------------------------------------------------------------------
    full_target = _concat_as_new_series([ts_target_train, ts_target_val, ts_target_test])
    full_covs = None
    if use_past and (ts_covs_train is not None) and (ts_covs_val is not None) and (ts_covs_test is not None):
        full_covs = _concat_as_new_series([ts_covs_train, ts_covs_val, ts_covs_test])
    
    # Boundaries (by position) on the fresh integer time axis
    start_val  = len(ts_target_train)
    start_test = len(ts_target_train) + len(ts_target_val)
    
    # Convert to the index values of the new full series
    start_val_time  = full_target.time_index[start_val]
    start_test_time = full_target.time_index[start_test]

    logging.info(
        "HF plan: len(full_target)=%d | start_val_idx=%d | start_test_idx=%d | horizon=%d | overlap_end=True",
        len(full_target),
        start_val,
        start_test,
        horizon,
    )

    logging.info(
        "Allowed last origin with overlap_end=False would be %d; with True we can go up to %d",
        len(full_target) - horizon,
        len(full_target) - 1,
    )


    def _hf(start_time):
        kwargs = dict(
            series=full_target,
            start=start_time,
            forecast_horizon=horizon,
            stride=1,
            retrain=False,
            verbose=False,
            last_points_only=False,
            show_warnings=False,
            overlap_end=True,
        )
        if use_past and full_covs is not None:
            kwargs["past_covariates"] = full_covs

        try:
            out = model.historical_forecasts(**kwargs)
            return out
        except Exception as e:
            _dbg("hf_error", model_key=model_key, error=str(e))
            raise

    
    pred_series_val_list  = _hf(start_val_time)
    pred_series_test_list = _hf(start_test_time)
    
    pred_val_np  = _stack_ribbons(pred_series_val_list)
    pred_test_np = _stack_ribbons(pred_series_test_list)


    return model, {}, pred_test_np, pred_val_np


