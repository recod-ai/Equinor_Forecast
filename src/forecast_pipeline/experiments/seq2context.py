# src/forecast_pipeline/experiments/seq2context.py

import logging
from data.data_loading import DataSource
from common.batch_preprocessing import load_and_preprocess_data
from common.seq_preprocessing import (
    prepare_data_seq,
)
from .base import BaseExperiment

class ExperimentSeq2Context(BaseExperiment):
    """
    Data provider for Seq2Context experiments.
    Only loads, filters, and packages inputs; no .run() here.
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
        feats = self.params.get(
            "selected_features",
            self.config.get("features", [main])
        )
        return main, feats

    def get_params(self):
        return self.params.copy()
    
    def run(self): # Adiciona type hint para corresponder à base
        """
        Implementação concreta do método 'run' exigido por BaseExperiment.
        Permite que a classe seja instanciada. O corpo está vazio ('pass')
        pois esta classe não é executada via 'run' neste fluxo de trabalho.
        """
        pass
    
    def load_and_prepare(self):
        """
        Load, preprocess, and prepare model inputs.

        This version is now architecture-aware, handling different configuration
        requirements for Seq2Context, Seq2PIN, etc., and is compatible with
        both legacy and profile-driven (HPO) execution modes.
        """
        logger = logging.getLogger(__name__)

        # 1. Retrieve parameters and load data (this part is unchanged)
        # -----------------------------------------------------------------
        p = self.get_params()
        main, feats = self.get_features()
        df = load_and_preprocess_data(DataSource, self.config, feats, self.well)

        aug_params = {"data_sample": p.get("data_sample", 0.5)}
        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler_X, scaler_target, y_train_original
        ) = prepare_data_seq(
            df, main, p["lag_window"], p["horizon"],
            test_size=p.get("test_size", 0.5), val_size=p.get("val_size", 0.1),
            data_aug_params=aug_params
        )

        # 2. Build the training arguments dictionary (train_kwargs)
        # -----------------------------------------------------------------

        # Get the specific architecture for this job to make decisions
        arch_name = p.get("architecture_name")

        # Start with the core data, which is common to all models
        train_kwargs = {
            "X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
        }

        # -----------------------------------------------------------------
        # NEW: expose natural reconstructed lengths for each split
        #      (N, H) -> length = N + H - 1, matching reconstruct_true_series
        # -----------------------------------------------------------------
        try:
            if y_val is not None and y_test is not None:
                N_val,  H_val  = y_val.shape
                N_test, H_test = y_test.shape

                # sanity check – in all current setups H_val == H_test == horizon
                if H_val != H_test:
                    logger.warning(
                        "seq2context_load_and_prepare: val/test horizons "
                        "do not match (H_val=%d, H_test=%d). Using test horizon.",
                        H_val, H_test,
                    )

                H = H_test
                T_val  = int(N_val  + H - 1)
                T_test = int(N_test + H - 1)

                train_kwargs["split_recon_lengths"] = {
                    "val":  T_val,
                    "test": T_test,
                }

                logger.info(
                    "seq2context_load_and_prepare: split_recon_lengths "
                    "val=%d test=%d (N_val=%d N_test=%d H=%d)",
                    T_val, T_test, N_val, N_test, H,
                )
            else:
                logger.info(
                    "seq2context_load_and_prepare: skip split_recon_lengths "
                    "reason=missing_y_val_or_y_test"
                )
        except Exception:
            # Defensive: never break data loading because of diagnostics
            logger.exception("seq2context_load_and_prepare: failed to infer split_recon_lengths")

        # --- Configuration Adapter Logic ---
        # This block ensures the correct config dictionaries are created and added.

        # a) Handle `strategy_config` (required by all Seq2* models)
        if "strategy_config" in p:
            # Legacy mode: The dictionary is already built.
            train_kwargs["strategy_config"] = p["strategy_config"]
        else:
            # Profile-driven mode: Build it from the flat key.
            if "physics_strategy" not in p:
                raise KeyError("'physics_strategy' is missing from the experiment profile.")
            train_kwargs["strategy_config"] = {"strategy_name": p["physics_strategy"]}

        # b) Handle `extractor_config` and `fuser_config` (only for specific architectures)
        if arch_name in ["Seq2Context", "Seq2Fuser"]:
            # These architectures require the data-driven components.
            if "extractor_config" in p and "fuser_config" in p:
                # Legacy mode: The dictionaries are already built.
                train_kwargs["extractor_config"] = p["extractor_config"]
                train_kwargs["fuser_config"] = p["fuser_config"]
            else:
                # Profile-driven mode: Get them from the expanded profile.
                if "extractor_config" not in p:
                    raise KeyError(f"'extractor_config' is required for {arch_name} but is missing.")
                if "fuser_config" not in p:
                    raise KeyError(f"'fuser_config' is required for {arch_name} but is missing.")
                train_kwargs["extractor_config"] = p["extractor_config"]
                train_kwargs["fuser_config"] = p["fuser_config"]

        # For other architectures like Seq2PIN or Seq2Trend, extractor/fuser configs are simply not added,
        # which is the correct behavior as they are not used by their respective create_model functions.

        # 3. Final return (unchanged)
        # -----------------------------------------------------------------
        prediction_input = X_test
        return train_kwargs, prediction_input, y_test, scaler_X, scaler_target, y_train_original
