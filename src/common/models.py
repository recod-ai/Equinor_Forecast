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


def train_deep_encoder_model(
    train_series: TimeSeries,
    train_covariates: TimeSeries,
    val_series: TimeSeries,
    val_covariates: TimeSeries,
    *,
    model_type: str = "NHiTS",
    output_chunk_length: int | str = 56,
) -> object:
    """
    Treina (e devolve) um modelo Darts escolhido por `model_type`.

    Agora, além dos “deep encoders” (“NHiTS”, “TiDE”, “TiDE+RIN”, “NLinear”,
    “N-Beats”), também suporta:

    - `"ARIMA"`        … ARIMA(p=1, d=1, q=1)          — linear clássico
    - `"AutoARIMA"`    … AutoArima() (busca automática)
    - `"LinearRegression"`
                       … Regressão linear multivariada (lags = input_chunk)

    Os novos modelos são treinados com **hiperparâmetros padrão**, garantindo
    pleno funcionamento no pipeline sem alterar a lógica downstream.
    """

    # --------------------- 1. parâmetros comuns aos “deep” ------------------
    optimizer_kwargs = {"lr": 1e-3}
    pl_trainer_kwargs = {
        "gradient_clip_val": 1,
        "max_epochs": 100,
        "accelerator": "auto",
        "callbacks": [],
    }
    lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
    lr_scheduler_kwargs = {"gamma": 0.999}

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-3,
        mode="min",
    )
    pl_trainer_kwargs["callbacks"] = [early_stopping]

    common_deep_args: Dict[str, Any] = dict(
        input_chunk_length=7,
        output_chunk_length=output_chunk_length,
        optimizer_kwargs=optimizer_kwargs,
        pl_trainer_kwargs=pl_trainer_kwargs,
        lr_scheduler_cls=lr_scheduler_cls,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        save_checkpoints=True,
        force_reset=True,
        batch_size=16,
        random_state=42,
    )

    # --------------------- 2. map de construtores ---------------------------
    def _nhits() -> NHiTSModel:
        return NHiTSModel(
            **common_deep_args,
            model_name="NHiTS",
            num_stacks=3,
            num_blocks=1,
            num_layers=2,
            layer_widths=512,
            dropout=0.1,
            activation="ReLU",
            MaxPool1d=True,
        )

    def _tide(use_rin: bool = False) -> TiDEModel:
        return TiDEModel(
            **common_deep_args,
            model_name="TiDE" + ("_RIN" if use_rin else ""),
            use_reversible_instance_norm=use_rin,
            num_encoder_layers=1,
            num_decoder_layers=1,
            decoder_output_dim=16,
            hidden_size=128,
            temporal_width_past=4,
            temporal_width_future=4,
            temporal_decoder_hidden=32,
            dropout=0.1,
        )

    def _nlinear() -> NLinearModel:
        return NLinearModel(
            **common_deep_args,
            model_name="NLinear",
            output_chunk_shift=0,
            shared_weights=False,
            const_init=True,
            normalize=False,
            use_static_covariates=True,
            n_epochs=20,
        )

    def _nbeats() -> NBEATSModel:
        return NBEATSModel(
            **common_deep_args,
            model_name="NBeats",
            generic_architecture=True,
            num_stacks=30,
            num_blocks=1,
            num_layers=4,
            layer_widths=512,
            dropout=0.1,
            activation="ReLU",
        )

    # ---- “clássicos” (não precisam de PyTorch nem validação) ---------------
    def _arima() -> ARIMA:
        return ARIMA(p=1, d=1, q=1)

    def _auto_arima() -> AutoARIMA:
        return AutoARIMA()

    def _linreg() -> LinearRegressionModel:
        # usa a janela de observação como lag
        lags = common_deep_args["input_chunk_length"]
        return LinearRegressionModel(
            lags=lags,
            output_chunk_length=output_chunk_length,
        )

    constructors: Dict[str, Callable[[], object]] = {
        # deep encoders
        "NHiTS": _nhits,
        "TiDE": lambda: _tide(False),
        "TiDE+RIN": lambda: _tide(True),
        "NLinear": _nlinear,
        "N-Beats": _nbeats,
        # clássicos
        "ARIMA": _arima,
        "AutoARIMA": _auto_arima,
        "LinearRegression": _linreg,
    }

    if model_type not in constructors:
        raise ValueError(f"Modelo desconhecido: {model_type}")

    model = constructors[model_type]()

    # --------------------- 3. treinamento -----------------------------------
    if model_type in {"NHiTS", "TiDE", "TiDE+RIN", "NLinear", "N-Beats"}:
        # deep encoder → com validação
        model.fit(
            series=train_series,
            val_series=val_series,
            # past_covariates=train_covariates,
            # val_past_covariates=val_covariates,
            verbose=False,
        )
    else:
        # clássicos → sem val_series (treino único)
        try:
            model.fit(
                series=train_series,
                # alguns modelos aceitam covariáveis → habilite se quiser
                # future_covariates=train_covariates,
                verbose=False,
            )
        except TypeError:
            # ARIMA/AutoARIMA não tem verbose
            model.fit(train_series)

    return model
