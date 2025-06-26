# SHAP Report – Iteration 500

    **Top 10 features**

    | Rank | Feature | Mean |SHAP| | Share (%) | Direction |
    |------|---------|-------------|-----------|-----------|
    |   rank | feature                  |   mean_|SHAP| |   relative_% | direction   |
|-------:|:-------------------------|--------------:|-------------:|:------------|
|      1 | BORE_OIL_VOL_MEAN_LAG_t0 |      0.253287 |     14.8132  | ↑           |
|      2 | BORE_OIL_VOL_LAG_7_t0    |      0.232463 |     13.5953  | ↓           |
|      3 | Prod_Start_Time_t0       |      0.218037 |     12.7517  | ↓           |
|      4 | BORE_OIL_VOL_LAG_6_t0    |      0.195082 |     11.4092  | ↓           |
|      5 | BORE_OIL_VOL_LAG_3_t0    |      0.151436 |      8.85656 | ↓           |
|      6 | BORE_OIL_VOL_LAG_2_t0    |      0.128457 |      7.5127  | ↑           |
|      7 | ON_STREAM_HRS_t0         |      0.122451 |      7.16145 | ↑           |
|      8 | BORE_OIL_VOL_LAG_1_t0    |      0.116374 |      6.80601 | ↑           |
|      9 | BORE_GAS_VOL_t0          |      0.113461 |      6.63565 | ↓           |
|     10 | BORE_OIL_VOL_LAG_5_t0    |      0.105368 |      6.16233 | ↓           |        # ← headers defaults to "keys"

    *Total importance is normalised to 100 %. Direction = monotonic sign of
    the SHAP dependence (Spearman).*  

    **Gini coefficient** of importance distribution: **0.836**