# src/hpo/grids_darts.py
from __future__ import annotations
from typing import Dict, List


def _with_profile(name: str, **kw) -> Dict:
    """Attach a human-readable profile name to a dict of hyperparams."""
    return {"profile": name, **kw}


def grid_tide(base: Dict) -> List[Dict]:
    """
    5 TiDE profiles, progressing from a shallow/fast model to a deep/complex one.
    Only includes kwargs accepted by TiDEModel.
    """
    return [
        _with_profile("tide_shallow_fast",
            hidden_size=64, num_encoder_layers=1, num_decoder_layers=1,
            decoder_output_dim=8, dropout=0.05, use_layer_norm=False),
        _with_profile("tide_shallow_reg",
            hidden_size=64, num_encoder_layers=1, num_decoder_layers=1,
            decoder_output_dim=16, dropout=0.2, use_layer_norm=True),
        _with_profile("tide_medium_ln",
            hidden_size=128, num_encoder_layers=2, num_decoder_layers=2,
            decoder_output_dim=32, dropout=0.1, use_layer_norm=True),
        _with_profile("tide_deep_ln",
            hidden_size=256, num_encoder_layers=3, num_decoder_layers=3,
            decoder_output_dim=64, dropout=0.1, use_layer_norm=True),
        _with_profile("tide_balanced",
            hidden_size=192, num_encoder_layers=2, num_decoder_layers=2,
            decoder_output_dim=32, dropout=0.15, use_layer_norm=True),
    ]


def grid_tide_rin(base: Dict) -> List[Dict]:
    """
    5 TiDE profiles for 'TiDE_RIN', biased towards regularization and stability.
    """
    return [
        _with_profile("tide_rin_shallow",
            hidden_size=64,  num_encoder_layers=1, num_decoder_layers=1,
            decoder_output_dim=16, dropout=0.2, use_layer_norm=True),
        _with_profile("tide_rin_medium",
            hidden_size=128, num_encoder_layers=2, num_decoder_layers=2,
            decoder_output_dim=32, dropout=0.25, use_layer_norm=True),
        _with_profile("tide_rin_balanced",
            hidden_size=160, num_encoder_layers=2, num_decoder_layers=2,
            decoder_output_dim=24, dropout=0.2, use_layer_norm=True),
        _with_profile("tide_rin_deep",
            hidden_size=256, num_encoder_layers=3, num_decoder_layers=3,
            decoder_output_dim=32, dropout=0.2, use_layer_norm=True),
        _with_profile("tide_rin_deep_reg",
            hidden_size=256, num_encoder_layers=3, num_decoder_layers=3,
            decoder_output_dim=64, dropout=0.3, use_layer_norm=True),
    ]


def grid_nhits(base: Dict) -> List[Dict]:
    """
    5 N-HiTS profiles, progressing from a tiny model to deep and wide variants.
    """
    return [
        _with_profile("nhits_tiny",        num_stacks=1, num_blocks=1, num_layers=1, layer_widths=64,  dropout=0.0),
        _with_profile("nhits_shallow",     num_stacks=2, num_blocks=1, num_layers=1, layer_widths=128, dropout=0.05),
        _with_profile("nhits_medium",      num_stacks=3, num_blocks=1, num_layers=2, layer_widths=256, dropout=0.1),
        _with_profile("nhits_deep",        num_stacks=4, num_blocks=2, num_layers=3, layer_widths=512, dropout=0.15),
        _with_profile("nhits_deep_reg",    num_stacks=4, num_blocks=3, num_layers=2, layer_widths=512, dropout=0.25),
    ]


def grid_nlinear(base: Dict) -> List[Dict]:
    """
    5 N-Linear profiles exploring key architectural choices like weight sharing.
    """
    return [
        _with_profile("nlinear_shared_nonorm",     shared_weights=True,  const_init=True,  normalize=False),
        _with_profile("nlinear_shared_norm",       shared_weights=True,  const_init=True,  normalize=True),
        _with_profile("nlinear_unshared_norm",     shared_weights=False, const_init=True,  normalize=True),
        _with_profile("nlinear_unshared_minimal",  shared_weights=False, const_init=False, normalize=False),
        _with_profile("nlinear_shared_strict",     shared_weights=True,  const_init=True,  normalize=True,  use_static_covariates=True),
    ]


def grid_linear_regression(base: Dict) -> List[Dict]:
    """
    5 LinearRegression profiles, varying lag length and the multi_models strategy.
    """
    L = base.get("input_chunk_length", 100)
    H = base.get("output_chunk_length", 100)
    return [
        _with_profile("linreg_lags_24",     lags=24,     output_chunk_length=H, multi_models=True),
        _with_profile("linreg_lags_L2",     lags=L//2,   output_chunk_length=H, multi_models=True),
        _with_profile("linreg_lags_L",      lags=L,      output_chunk_length=H, multi_models=True),
        _with_profile("linreg_lags_24_mmF", lags=24,     output_chunk_length=H, multi_models=False),
        _with_profile("linreg_lags_L_mmF",  lags=L,      output_chunk_length=H, multi_models=False),
    ]


def grid_arima(base: Dict) -> List[Dict]:
    """
    5 ARIMA profiles, from simple non-seasonal orders to basic seasonal models.
    """
    return [
        _with_profile("arima_110_ns", p=1, d=1, q=0, seasonal_order=(0,0,0,0)),
        _with_profile("arima_012_ns", p=0, d=1, q=2, seasonal_order=(0,0,0,0)),
        _with_profile("arima_111_ns", p=1, d=1, q=1, seasonal_order=(0,0,0,0)),
        _with_profile("sarima_011_7", p=0, d=1, q=1, seasonal_order=(0,1,1,7)),
        _with_profile("sarima_111_12", p=1, d=1, q=1, seasonal_order=(1,1,1,12)),
    ]


def grid_autoarima(base: Dict) -> List[Dict]:
    """
    5 AutoARIMA profiles, varying search strategy (stepwise) and seasonality.
    """
    return [
        _with_profile("autoarima_fast_ns",   seasonal=False, m=0,  stepwise=True,  max_p=3, max_q=3),
        _with_profile("autoarima_wide_ns",   seasonal=False, m=0,  stepwise=False, max_p=5, max_q=5),
        _with_profile("autoarima_fast_7",    seasonal=True,  m=7,  stepwise=True,  max_p=3, max_q=3, max_P=1, max_Q=1),
        _with_profile("autoarima_fast_12",   seasonal=True,  m=12, stepwise=True,  max_p=3, max_q=3, max_P=1, max_Q=1),
        _with_profile("autoarima_wide_7",    seasonal=True,  m=7,  stepwise=False, max_p=5, max_q=5, max_P=2, max_Q=2),
    ]


# ---- public dispatcher ------------------------------------------------------ #

GRID_BUILDERS = {
    "TiDE": grid_tide,
    "TiDE_RIN": grid_tide_rin,
    "NHiTS": grid_nhits,
    "NLinear": grid_nlinear,
    "LinearRegression": grid_linear_regression,
    "ARIMA": grid_arima,
    "AutoARIMA": grid_autoarima,
    # "N-Beats":  # intentionally omitted (computationally expensive)
}

def make_search_grid(model_key: str, base_params: Dict) -> List[Dict]:
    """
    Return a list of ~5 candidate profiles for the given model_key.
    Raises KeyError if the model_key isn't supported.
    """
    return GRID_BUILDERS[model_key](base_params)