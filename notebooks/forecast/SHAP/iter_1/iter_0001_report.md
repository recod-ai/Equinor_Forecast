# SHAP Report – Iteration 1

    **Top 10 features**

    | Rank | Feature | Mean |SHAP| | Share (%) | Direction |
    |------|---------|-------------|-----------|-----------|
    |   rank | feature                  |   mean_|SHAP| |   relative_% | direction   |
|-------:|:-------------------------|--------------:|-------------:|:------------|
|      1 | Prod_Start_Time_t0       |      0.487724 |     16.8177  | ↓           |
|      2 | BORE_OIL_VOL_LAG_1_t0    |      0.470156 |     16.2119  | ↓           |
|      3 | BORE_OIL_VOL_LAG_7_t0    |      0.382628 |     13.1938  | ↓           |
|      4 | BORE_OIL_VOL_LAG_2_t0    |      0.30045  |     10.3601  | ↑           |
|      5 | BORE_GAS_VOL_t0          |      0.229349 |      7.90842 | ↑           |
|      6 | ON_STREAM_HRS_t0         |      0.227469 |      7.84359 | ↑           |
|      7 | BORE_OIL_VOL_LAG_3_t0    |      0.213144 |      7.34963 | ↓           |
|      8 | BORE_OIL_VOL_LAG_4_t0    |      0.172254 |      5.93966 | ↓           |
|      9 | BORE_OIL_VOL_MEAN_LAG_t0 |      0.147668 |      5.09188 | ↑           |
|     10 | BORE_OIL_VOL_LAG_5_t0    |      0.136458 |      4.70534 | ↓           |        # ← headers defaults to "keys"

    *Total importance is normalised to 100 %. Direction = monotonic sign of
    the SHAP dependence (Spearman).*  

    **Gini coefficient** of importance distribution: **0.838**