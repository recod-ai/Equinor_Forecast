import logging
from data.data_loading import DataSource
from common.batch_preprocessing import load_and_preprocess_data
from common.seq_preprocessing import prepare_data_seq
from .base import BaseExperiment

class ExperimentArps(BaseExperiment):
    """
    Data provider for the Arps backend.
    Phase 0: only loads and prepares data; no effective .run() (uses an external runner).
    """
    def __init__(self, config, well, params, exp_id):
        self.config = config
        self.well = well
        self.params = params
        self.exp_id = exp_id
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Not used in this flow (the runner calls load_and_prepare + external trainer)
        pass

    def _get_features(self):
        main = self.config.get("target_column", self.config["load_params"].get("series_name"))
        feats = self.params.get("selected_features", self.config.get("features", [main]))
        return main, feats

    def load_and_prepare(self):
        p = self.params.copy()
        main, feats = self._get_features()
        df = load_and_preprocess_data(DataSource, self.config, feats, self.well)

        aug_params = {"data_sample": p.get("data_sample", 1.0)}
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         scaler_X, scaler_target, y_train_original) = prepare_data_seq(
            df, main, p["lag_window"], p["horizon"],
            test_size=p.get("test_size", 0.5), val_size=p.get("val_size", 0.1),
            data_aug_params=aug_params
        )

        # Phase 0: we return everything the runner needs
        train_kwargs = {
            "X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
            # config placeholders; will be used in the next phases
            "arps_cfg": p.get("arps", {}),            # optional (if it comes from the profile)
            "aggregation_method": p.get("aggregation_method", "reconstruct"),
        }
        prediction_input = X_test
        return train_kwargs, prediction_input, y_test, scaler_X, scaler_target, y_train_original