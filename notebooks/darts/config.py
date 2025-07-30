# File: config.py

from typing import Dict, Any

# ==============================================================================
# Central Hyperparameter Configuration for All Models
# ==============================================================================
# This file defines the different hyperparameter sets for each model architecture.
# "Medium (Balanced)" contains the original default parameters for reproducibility.
# ==============================================================================

HYPERPARAM_DESCRIPTIONS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "NHiTS": {
        "Small & Fast": {
            "num_stacks": 2,
            "num_blocks": 1,
            "num_layers": 2,
            "layer_widths": 256,
            "dropout": 0.1,
            "activation": "ReLU",
            "MaxPool1d": False,
        },
        "Medium (Balanced)": {
            "num_stacks": 3,
            "num_blocks": 1,
            "num_layers": 2,
            "layer_widths": 512,
            "dropout": 0.1,
            "activation": "ReLU",
            "MaxPool1d": True,
        },
        "Large & Robust": {
            "num_stacks": 5,
            "num_blocks": 2,
            "num_layers": 4,
            "layer_widths": 512,
            "dropout": 0.2,
            "activation": "ReLU",
            "MaxPool1d": True,
        },
        "Stable & Regularized": {
            "num_stacks": 3,
            "num_blocks": 1,
            "num_layers": 2,
            "layer_widths": 512,
            "dropout": 0.4,
            "activation": "ReLU",
            "MaxPool1d": True,
        },
        "Wide & Shallow": {
            "num_stacks": 2,
            "num_blocks": 1,
            "num_layers": 1,
            "layer_widths": 1024,
            "dropout": 0.2,
            "activation": "ReLU",
            "MaxPool1d": True,
        },
    },
    "TiDE": {
        "Small & Fast": {
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "decoder_output_dim": 8,
            "hidden_size": 64,
            "temporal_width_past": 2,
            "temporal_width_future": 2,
            "temporal_decoder_hidden": 16,
            "dropout": 0.1,
        },
        "Medium (Balanced)": {
            "use_reversible_instance_norm": False,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "decoder_output_dim": 16,
            "hidden_size": 128,
            "temporal_width_past": 4,
            "temporal_width_future": 4,
            "temporal_decoder_hidden": 32,
            "dropout": 0.1,
        },
        "Large & Robust": {
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "decoder_output_dim": 32,
            "hidden_size": 256,
            "temporal_width_past": 4,
            "temporal_width_future": 4,
            "temporal_decoder_hidden": 64,
            "dropout": 0.2,
        },
        "Stable & Regularized": {
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "decoder_output_dim": 16,
            "hidden_size": 128,
            "temporal_width_past": 4,
            "temporal_width_future": 4,
            "temporal_decoder_hidden": 32,
            "dropout": 0.5,
        },
        "Wide & Shallow": {
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "decoder_output_dim": 16,
            "hidden_size": 512,
            "temporal_width_past": 2,
            "temporal_width_future": 2,
            "temporal_decoder_hidden": 16,
            "dropout": 0.25,
        },
    },
    "TiDE+RIN": {
        "Small & Fast": {
            "use_reversible_instance_norm": True,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "decoder_output_dim": 8,
            "hidden_size": 64,
            "dropout": 0.1,
        },
        "Medium (Balanced)": {
            "use_reversible_instance_norm": True,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "decoder_output_dim": 16,
            "hidden_size": 128,
            "temporal_width_past": 4,
            "temporal_width_future": 4,
            "temporal_decoder_hidden": 32,
            "dropout": 0.1,
        },
        "Large & Robust": {
            "use_reversible_instance_norm": True,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "decoder_output_dim": 32,
            "hidden_size": 256,
            "dropout": 0.2,
        },
        "Stable & Regularized": {
            "use_reversible_instance_norm": True,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "decoder_output_dim": 16,
            "hidden_size": 128,
            "dropout": 0.5,
        },
        "Wide & Shallow": {
            "use_reversible_instance_norm": True,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "decoder_output_dim": 16,
            "hidden_size": 512,
            "dropout": 0.25,
        },
    },
    "NLinear": {
        "Small & Fast": {
            "shared_weights": True,
            "const_init": True,
            "normalize": False,
            "n_epochs": 50,
        },
        "Medium (Balanced)": {
            "shared_weights": False,
            "const_init": True,
            "normalize": False,
            "use_static_covariates": True,
        },
        "Large & Robust": {
            "shared_weights": False,
            "const_init": False,
            "normalize": False,
        },
        "Stable & Regularized": {
            "shared_weights": False,
            "const_init": True,
            "use_static_covariates": True,
            "use_reversible_instance_norm": True,
            "n_epochs": 50,
        },
        "Wide & Shallow": {
            "shared_weights": False,
            "const_init": False,
            "normalize": False,
        },
    },
    "N-Beats": {
        "Small & Fast": {
            "num_stacks": 10,
            "num_blocks": 1,
            "num_layers": 2,
            "layer_widths": 256,
            "dropout": 0.1,
            "activation": "ReLU",
        },
        "Medium (Balanced)": {
            "generic_architecture": True,
            "num_stacks": 30,
            "num_blocks": 1,
            "num_layers": 4,
            "layer_widths": 512,
            "dropout": 0.1,
            "activation": "ReLU",
        },
        "Large & Robust": {
            "generic_architecture": True,
            "num_stacks": 40,
            "num_blocks": 2,
            "num_layers": 4,
            "layer_widths": 1024,
            "dropout": 0.2,
            "activation": "ReLU",
        },
        "Stable & Regularized": {
            "generic_architecture": True,
            "num_stacks": 30,
            "num_blocks": 1,
            "num_layers": 4,
            "layer_widths": 512,
            "dropout": 0.5,
            "activation": "ReLU",
        },
        "Wide & Shallow": {
            "generic_architecture": True,
            "num_stacks": 15,
            "num_blocks": 1,
            "num_layers": 2,
            "layer_widths": 1024,
            "dropout": 0.25,
            "activation": "ReLU",
        },
    },

    "ARIMA": {
    # Simple, fast model, little memory of the past
    "Small & Fast": {
        "p": 1,
        "d": 0,
        "q": 0,
        "trend": "n",  # No trend
    },
    # Balanced: your default config
    "Medium (Balanced)": {
        "p": 1,
        "d": 1,
        "q": 1,
        "trend": "n",  # Constant trend
    },
    # Large, tries to capture more complexity, allows trend and seasonality
    "Large & Robust": {
        "p": 2,
        "d": 2,
        "q": 2,
        "seasonal_order": (1, 1, 1, 7), 
        "trend": "n",  
    },
    # Strong regularization by limiting trend and seasonality, avoids overfitting
    "Stable & Regularized": {
        "p": 1,
        "d": 0,
        "q": 2,
        "seasonal_order": (0, 0, 0, 0),  # No seasonality
        "trend": "c", 
    },
    # Wide (complex lags) but shallow (no moving average, no trend)
    "Wide & Shallow": {
        "p": [1, 2, 3, 4, 5],  # Custom AR lags, broad memory
        "d": 0,
        "q": 0,
        "trend": "n",
    },
    },

    "LinearRegression": {
    # Fastest: deterministic, single model per step, no quantiles
    "Small & Fast": {
        "multi_models": True,
        "likelihood": None,
        "quantiles": None,
        "random_state": 7,
    },
    # Default: probabilistic, quantile regression, multi_models
    "Medium (Balanced)": {
        "multi_models": True,
        "likelihood": "quantile",
        "quantiles": [0.1, 0.5, 0.9],
        "random_state": 21,
    },
    # Robust: probabilistic, quantile regression, all steps predicted together (multioutput)
    "Large & Robust": {
        "multi_models": False,
        "likelihood": "quantile",
        "quantiles": [0.1, 0.5, 0.9],
        "random_state": 35,
    },
    # Regularized: deterministic, multi_models, still reproducible
    "Stable & Regularized": {
        "multi_models": True,
        "likelihood": None,
        "quantiles": None,
        "random_state": 17,
    },
    # Wide & Shallow: probabilistic, multi_models, with more quantiles (example of different quantiles)
    "Wide & Shallow": {
        "multi_models": True,
        "likelihood": "quantile",
        "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
        "random_state": 26,
    },
    },

        "AutoARIMA": {
        "Medium (Balanced)": {}
    },

}

# Human-readable descriptions for generating tables in your paper.
CONFIG_DESCRIPTIONS: Dict[str, str] = {
    "Small & Fast": "A lightweight configuration optimized for speed.",
    "Medium (Balanced)": "A default, balanced configuration for baseline results.",
    "Large & Robust": "A larger model designed for higher accuracy.",
    "Stable & Regularized": "A configuration focused on high regularization to prevent overfitting.",
    "Wide & Shallow": "A configuration using wide but shallow layers to capture patterns without excessive depth.",
}


