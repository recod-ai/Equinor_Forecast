# import os, matplotlib
# os.environ.setdefault("MPLBACKEND", "Agg")
# matplotlib.use("Agg")


# ---------- standard library ----------
import logging
import textwrap
from pathlib import Path

# ---------- third-party ----------
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense

# ---------- project (relativos a partir de src) ----------
from data.data_loading import DataSource
from common.batch_preprocessing import load_and_preprocess_data
from common.batch_preprocessing import (load_and_preprocess_data,
                                            prepare_data,
                                            prepare_inputs)


from common.seq_preprocessing import (
    apply_filter_to_X_and_y,
)

from evaluation.evaluation import (
    plot_results,
    evaluate_model,
    evaluate_cumulative,
    display_metrics,
    compute_metrics_to_df,
    display_data_overview,
)

from .base import BaseExperiment

import logging
import numpy as np


class ExperimentSeq2Value(BaseExperiment):
    """
    Simplified Data Provider for Seq2Value experiments.
    Only handles data loading, filtering, and input packaging.
    """

    def __init__(self, config, well, params, exp_id):
        self.config = config
        self.well = well
        self.params = params
        self.exp_id = exp_id

    def get_features(self):
        main = self.config.get(
            "target_column",
            self.config["load_params"].get("serie_name")
        )
        feats = ['PI', 'BORE_GAS_VOL', 'BORE_OIL_VOL']
        return main, feats

    def get_params(self):
        return self.params.copy()

    def run(self):
        """
        Concrete but empty 'run' method required by BaseExperiment.
        Actual execution logic handled externally in the pipeline.
        """
        pass

    def load_and_prepare(self):
        """
        Loads data, applies preprocessing, optional filtering,
        and prepares model inputs.
        Returns:
          train_kwargs, prediction_input, y_test, scaler_target, y_train_orig
        """
        p = self.get_params()
        main, feats = self.get_features()

        # Load and preprocess data
        df = load_and_preprocess_data(DataSource, self.config, feats, self.well)
        X_tr, X_te, y_tr, y_te, scaler, y_tr_orig = prepare_data(
            df, main, p["lag_window"], p["horizon"], p["test_size"], data_aug=True
        )

        # Optional adaptive filtering
        if p["apply_adaptive_filtering"]:
            X_tr, y_tr = apply_filter_to_X_and_y(
                X_tr, y_tr, method=p["filter_method"], **p["filter_kwargs"]
            )
            X_te, y_te = apply_filter_to_X_and_y(
                X_te, y_te, method=p["filter_method"], **p["filter_kwargs"]
            )

        # Prepare network inputs
        train_kwargs, pred_input = prepare_inputs(
            X_tr, X_te, y_tr, y_te,
            p["feature_kind"], feats, main
        )

        # Propagate model configs
        train_kwargs.update({
            "strategy_config": p["strategy_config"],
            "extractor_config": p["extractor_config"],
            "fuser_config": p["fuser_config"],
        })

        return train_kwargs, pred_input, y_te, scaler, y_tr_orig



