import logging
from data.data_loading import DataSource
from common.batch_preprocessing import load_and_preprocess_data
from common.seq_preprocessing import (
    prepare_data_seq,
    apply_filter_to_X_and_y,
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
        if self.params.get("use_known_good", False):
            feats = ['BORE_GAS_VOL','CE','PI','BORE_OIL_VOL_Original', main]
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
        Load, preprocess, optional filter, and prepare model inputs.
    
        Returns:
          train_kwargs: dict containing training and validation data plus config.
          prediction_input: test input data for inference.
          y_test: test targets (scaled).
          scaler_target: scaler for the target variable.
          y_train_original: original unscaled training targets.
        """
        # 1) Retrieve parameters and feature definitions
        p = self.get_params()
        main, feats = self.get_features()

        # 2) Load and preprocess data, then split into train, validation, and test
        df = load_and_preprocess_data(DataSource, self.config, feats, self.well)

        # logging.info(df.head())
        # logging.info(df.describe())
        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler_target,
            y_train_original
        ) = prepare_data_seq(
            df,
            main,
            p["lag_window"],
            p["horizon"],
            test_size=p.get("test_size", 0.6),
            val_size=p.get("val_size", 0.1),
            data_aug=True
        )
    
        # 3) Optional adaptive filtering on train, validation, and test sets
        if p.get("apply_adaptive_filtering", False):
            X_train, y_train = apply_filter_to_X_and_y(
                X_train, y_train,
                method=p["filter_method"],
                **p.get("filter_kwargs", {})
            )
            X_val, y_val = apply_filter_to_X_and_y(
                X_val, y_val,
                method=p["filter_method"],
                **p.get("filter_kwargs", {})
            )
            X_test, y_test = apply_filter_to_X_and_y(
                X_test, y_test,
                method=p["filter_method"],
                **p.get("filter_kwargs", {})
            )
    
        # 4) Prepare train_kwargs and prediction_input directly
        train_kwargs = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val":   X_val,
            "y_val":   y_val,
            "strategy_config": p["strategy_config"],
            "extractor_config": p["extractor_config"],
            "fuser_config": p["fuser_config"]
        }
        prediction_input = X_test

        
    
        return train_kwargs, prediction_input, y_test, scaler_target, y_train_original

