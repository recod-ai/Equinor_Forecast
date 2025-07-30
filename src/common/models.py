# --------------------------------------------------------------------------- #
# train_deep_encoder_model – agora aceita “Deep Learning” *e* “Clássicos”     #
# --------------------------------------------------------------------------- #
from typing import Dict, Callable, Any

import torch
from darts import TimeSeries
from darts.models import (
    # deep encoders que você já usava
    NHiTSModel, TiDEModel, NLinearModel, NBEATSModel,
    # novos modelos “clássicos”
    ARIMA, AutoARIMA, LinearRegressionModel,
)
from pytorch_lightning.callbacks import EarlyStopping

# In your `models.py` file, replace the old function with this one.

from typing import Dict, Any
import torch
from darts import TimeSeries
from darts.models import (
    NHiTSModel, TiDEModel, NLinearModel, NBEATSModel,
    ARIMA, AutoARIMA, LinearRegressionModel
)
from pytorch_lightning.callbacks import EarlyStopping

def train_deep_encoder_model(
    train_series: TimeSeries,
    val_series: TimeSeries,
    model_type: str,
    output_chunk_length: int,
    hyperparam_config: Dict[str, Any],  # <-- THE CRUCIAL PARAMETER
    train_covariates: TimeSeries = None,
    val_covariates: TimeSeries = None,
) -> object:
    """
    Trains a Darts model chosen by `model_type`, dynamically configured by `hyperparam_config`.
    This version is driven by the configuration file and is highly flexible.
    """
    print(f"  > Training model: {model_type} with config: {hyperparam_config}")
    
    # Allow accelerator and batch_size to be configured, with safe defaults.
    accelerator = hyperparam_config.pop("accelerator", "cpu")
    batch_size = hyperparam_config.pop("batch_size", 16)

    # Common arguments for all deep learning models
    common_deep_args = {
        "input_chunk_length": 7,
        "output_chunk_length": output_chunk_length,
        "optimizer_kwargs": {"lr": 1e-3},
        "pl_trainer_kwargs": {
            "gradient_clip_val": 1,
            "max_epochs": 100,
            "accelerator": accelerator,
            "callbacks": [EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-3, mode="min")],
        },
        "lr_scheduler_cls": torch.optim.lr_scheduler.ExponentialLR,
        "lr_scheduler_kwargs": {"gamma": 0.999},
        "save_checkpoints": False,        # <= NÃO salva checkpoints!
        "log_tensorboard": False,         # <= NÃO gera logs!
        "work_dir": "/tmp/darts_noop", 
        "force_reset": True,
        "batch_size": batch_size,
        "random_state": 42,
        "save_checkpoints": False,
        "force_reset": True,
        
    }

    # Map model type string to Darts model class
    model_constructors = {
        "NHiTS": NHiTSModel, "TiDE": TiDEModel, "TiDE+RIN": TiDEModel,
        "N-Beats": NBEATSModel, "NLinear": NLinearModel, "ARIMA": ARIMA,
        "AutoARIMA": AutoARIMA, "LinearRegression": LinearRegressionModel,
    }

    if model_type not in model_constructors:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    ModelClass = model_constructors[model_type]
    is_deep_learning_model = model_type in {"NHiTS", "TiDE", "TiDE+RIN", "N-Beats", "NLinear"}

    # Instantiate the model with the correct parameters
    if is_deep_learning_model:
        # For deep learning, merge common args with specific hyperparams.
        # The specific hyperparams from the config file will overwrite the defaults.
        model_params = {**common_deep_args, **hyperparam_config}
        model = ModelClass(**model_params)
    else:
        # For classical models, just use the hyperparams from the config.
        model_params = hyperparam_config.copy()
        if model_type == "LinearRegression":
             model_params['lags'] = common_deep_args['input_chunk_length']
             model_params['output_chunk_length'] = output_chunk_length
        model = ModelClass(**model_params)

    # --- Training Logic ---
    if is_deep_learning_model:
        model.fit(
            series=train_series,
            val_series=val_series,
            # past_covariates=train_covariates,
            # val_past_covariates=val_covariates,
            verbose=False,
            dataloader_kwargs={"num_workers": 1}
        )
    else:
        # Classical models do not use a validation set
        model.fit(series=train_series)

    return model