# SHAP Report – Iteration 1500

    **Top 10 features**

    | Rank | Feature | Mean |SHAP| | Share (%) | Direction |
    |------|---------|-------------|-----------|-----------|
    |   rank | feature                  |   mean_|SHAP| |   relative_% | direction   |
|-------:|:-------------------------|--------------:|-------------:|:------------|
|      1 | BORE_GAS_VOL_t0          |     0.129158  |     17.4298  | ↓           |
|      2 | BORE_OIL_VOL_MEAN_LAG_t0 |     0.105341  |     14.2158  | ↑           |
|      3 | BORE_OIL_VOL_LAG_6_t0    |     0.0794769 |     10.7254  | ↓           |
|      4 | BORE_OIL_VOL_LAG_5_t0    |     0.0762024 |     10.2835  | ↓           |
|      5 | BORE_OIL_VOL_LAG_1_t0    |     0.0726009 |      9.79746 | ↑           |
|      6 | Prod_Start_Time_t0       |     0.0705918 |      9.52633 | ↑           |
|      7 | BORE_OIL_VOL_LAG_3_t0    |     0.0608714 |      8.21457 | ↓           |
|      8 | BORE_OIL_VOL_LAG_4_t0    |     0.0501068 |      6.7619  | ↑           |
|      9 | BORE_OIL_VOL_LAG_2_t0    |     0.043708  |      5.89838 | ↑           |
|     10 | BORE_OIL_VOL_LAG_7_t0    |     0.0310485 |      4.18998 | ↑           |        # ← headers defaults to "keys"

    *Total importance is normalised to 100 %. Direction = monotonic sign of
    the SHAP dependence (Spearman).*  

    **Gini coefficient** of importance distribution: **0.837**