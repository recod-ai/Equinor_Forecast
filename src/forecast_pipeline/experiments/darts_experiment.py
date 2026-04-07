# src/forecast_pipeline/experiments/darts.py

# 1. Standard library imports
from typing import Any, Dict, List, Tuple

# 2. Third-party imports
import numpy as np
import pandas as pd
from darts import TimeSeries

# 3. Local application imports
from common.batch_preprocessing import load_and_preprocess_data
from common.seq_preprocessing import prepare_data_seq
from data.data_loading import DataSource
from sklearn.preprocessing import RobustScaler, StandardScaler

from .base import BaseExperiment

import pandas as pd
import numpy as np
from darts import TimeSeries

from common.seq_preprocessing import prepare_data_for_darts, report_split_sample_counts

def _normalize_darts_params(p: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(p)
    # accept old names too
    q.setdefault("input_chunk_length", q.get("lag_window"))
    q.setdefault("output_chunk_length", q.get("horizon"))
    return q

class ExperimentDarts(BaseExperiment):
    def __init__(self, config, well, params, exp_id):
        super().__init__(ds=config, well=well, params=params, exp_id=exp_id)

    def get_features(self) -> Tuple[str, List[str]]:
        main = self.ds.get("target_column")
        feats = self.params.get("selected_features", self.ds.get("features", [main]))
        return main, feats

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def run(self): # Adiciona type hint para corresponder à base
        """
        Implementação concreta do método 'run' exigido por BaseExperiment.
        Permite que a classe seja instanciada. O corpo está vazio ('pass')
        pois esta classe não é executada via 'run' neste fluxo de trabalho.
        """
        pass

    def load_and_prepare(self) -> Tuple:
        p = _normalize_darts_params(self.get_params())
        main_col, feature_cols = self.get_features()

        # 1) Load flat DF with the trusted loader
        df = load_and_preprocess_data(DataSource, self.ds, feature_cols, self.well)

        # 2) Continuous chronological split + scaling
        (df_train_s, df_val_s, df_test_s,
         scaler_X, scaler_target,
         y_train_unscaled, y_test_unscaled) = prepare_data_for_darts(
            df,
            target_col=main_col,
            test_size=p.get("test_size", 0.5),
            val_size=p.get("val_size", 0.1),
            scaler_type=p.get("scaler_type", "robust")
        )

        # 3) Give each split a contiguous time index (daily is fine)
        full_idx = pd.date_range(start="2000-01-01", periods=len(df), freq="D")
        df_train_s.index = full_idx[:len(df_train_s)]
        df_val_s.index   = full_idx[len(df_train_s):len(df_train_s)+len(df_val_s)]
        df_test_s.index  = full_idx[len(df_train_s)+len(df_val_s):]

        # 4) Build multivariate TimeSeries for Darts
        ts_train = TimeSeries.from_dataframe(df_train_s, fill_missing_dates=True, freq="D")
        ts_val   = TimeSeries.from_dataframe(df_val_s,   fill_missing_dates=True, freq="D")
        ts_test  = TimeSeries.from_dataframe(df_test_s,  fill_missing_dates=True, freq="D")

        report_split_sample_counts(
            train_len=len(ts_train),
            val_len=len(ts_val),
            test_len=len(ts_test),
            input_length=self.params.get('input_chunk_length'),
            output_length=self.params.get('output_chunk_length'),
            backend='Darts'
        )

        # 5) “Briefcase” keyed exactly like your current trainer expects
        train_kwargs = {
            "X_train": ts_train,
            "X_val":   ts_val,
            "main_col": main_col,
            "params": p,
            # plot needs these scalers
            "scaler_X": scaler_X,
            "scaler_target": scaler_target,
            "dataset_name": self.ds["name"],
        }
        prediction_input = {"ts_test": ts_test}

        # Return the same 6-tuple your gateway expects
        return (
            train_kwargs,              # 0
            prediction_input,          # 1
            y_test_unscaled.values,    # 2
            scaler_X,                  # 3
            scaler_target,             # 4
            y_train_unscaled.values    # 5
        )


