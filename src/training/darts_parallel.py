import os
import torch
from typing import List, Tuple
from darts import TimeSeries
from darts.models import TiDEModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def process_window(
    window_start: int,
    full_series: TimeSeries,
    full_covariates: TimeSeries,
    forecast_horizon: int,
    initial_train_size: int,
    model_types: List[str]
) -> Tuple[int, TimeSeries]:
    """
    Processa uma única janela de tempo, treinando o modelo e realizando previsões.
    """
    window_end = window_start + forecast_horizon
    train_series = full_series[:window_start]
    test_series = full_series[window_start:window_end]
    covariates_train = full_covariates[:window_start]
    covariates_test = full_covariates[window_start:window_end]

    train, val = train_series.split_after(0.7)
    train_covariates = covariates_train.split_after(0.7)[0]
    val_covariates = covariates_train.split_after(0.7)[1]
    
    print('aqui')

    # Treinamento do modelo
    model = train_deep_encoder_model(
        train_series=train,
        train_covariates=train_covariates,
        val_series=val,
        val_covariates=val_covariates,
        model_type=model_types[0]  # Supondo um único modelo
    )
    
    print('ali')

    forecast = model.predict(n=forecast_horizon, series=train_series, verbose=False)
    forecast_aligned = forecast.slice_intersect(test_series)

    if window_start == initial_train_size:
        return window_start, forecast_aligned
    else:
        return window_start, forecast_aligned[-1:]

def train_deep_encoder_model(
    train_series: TimeSeries,
    train_covariates: TimeSeries,
    val_series: TimeSeries,
    val_covariates: TimeSeries,
    model_type: str
) -> object:
    """
    Treina um modelo de aprendizado profundo com covariáveis temporais.
    """
    optimizer_kwargs = {"lr": 1e-3}
    pl_trainer_kwargs = {
        "gradient_clip_val": 1,
        "max_epochs": 25,
        "accelerator": "cpu",
        "callbacks": [EarlyStopping(monitor="val_loss", patience=25, min_delta=1e-3)],
    }
    lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
    lr_scheduler_kwargs = {"gamma": 0.999}

    common_model_args = {
        "input_chunk_length": 7,
        "output_chunk_length": 56,
        "optimizer_kwargs": optimizer_kwargs,
        "pl_trainer_kwargs": pl_trainer_kwargs,
        "lr_scheduler_cls": lr_scheduler_cls,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
        "batch_size": 16,
        "random_state": 42,
    }

    if model_type == "TiDE":
        model = TiDEModel(
            **common_model_args,
            model_name="TiDE_Model",
            use_reversible_instance_norm=False,
            num_encoder_layers=2,
            num_decoder_layers=2,
            decoder_output_dim=32,
            hidden_size=128,
            temporal_width_past=4,
            temporal_width_future=4,
            use_layer_norm=True,
            dropout=0.1,
        )
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")

    model.fit(
        series=train_series,
        val_series=val_series,
        verbose=True,
    )
    return model
