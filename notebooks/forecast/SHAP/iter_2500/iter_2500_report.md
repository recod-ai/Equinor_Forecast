# SHAP Report – Iteration 2500

    **Top 10 features**

    | Rank | Feature | Mean |SHAP| | Share (%) | Direction |
    |------|---------|-------------|-----------|-----------|
    |   rank | feature                  |   mean_|SHAP| |   relative_% | direction   |
|-------:|:-------------------------|--------------:|-------------:|:------------|
|      1 | BORE_OIL_VOL_LAG_5_t0    |     0.166625  |     21.1857  | ↓           |
|      2 | BORE_OIL_VOL_LAG_6_t0    |     0.140846  |     17.908   | ↓           |
|      3 | BORE_OIL_VOL_LAG_1_t0    |     0.0916619 |     11.6545  | ↑           |
|      4 | BORE_OIL_VOL_MEAN_LAG_t0 |     0.0791342 |     10.0616  | ↑           |
|      5 | BORE_OIL_VOL_LAG_7_t0    |     0.0689045 |      8.76097 | ↑           |
|      6 | Prod_Start_Time_t0       |     0.0514997 |      6.548   | ↑           |
|      7 | BORE_OIL_VOL_LAG_4_t0    |     0.0502268 |      6.38616 | ↑           |
|      8 | BORE_GAS_VOL_t0          |     0.0491477 |      6.24896 | ↓           |
|      9 | ON_STREAM_HRS_t0         |     0.0421759 |      5.36252 | ↑           |
|     10 | BORE_OIL_VOL_LAG_2_t0    |     0.0261319 |      3.32258 | ↓           |        # ← headers defaults to "keys"

    *Total importance is normalised to 100 %. Direction = monotonic sign of
    the SHAP dependence (Spearman).*  

    **Gini coefficient** of importance distribution: **0.840**