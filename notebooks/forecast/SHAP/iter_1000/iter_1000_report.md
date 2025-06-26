# SHAP Report – Iteration 1000

    **Top 10 features**

    | Rank | Feature | Mean |SHAP| | Share (%) | Direction |
    |------|---------|-------------|-----------|-----------|
    |   rank | feature                  |   mean_|SHAP| |   relative_% | direction   |
|-------:|:-------------------------|--------------:|-------------:|:------------|
|      1 | BORE_GAS_VOL_t0          |     0.158458  |     19.528   | ↓           |
|      2 | BORE_OIL_VOL_MEAN_LAG_t0 |     0.113769  |     14.0206  | ↑           |
|      3 | BORE_OIL_VOL_LAG_4_t0    |     0.0935306 |     11.5265  | ↑           |
|      4 | BORE_OIL_VOL_LAG_5_t0    |     0.0919105 |     11.3269  | ↓           |
|      5 | BORE_OIL_VOL_LAG_6_t0    |     0.0875982 |     10.7954  | ↓           |
|      6 | Prod_Start_Time_t0       |     0.0823083 |     10.1435  | ↑           |
|      7 | ON_STREAM_HRS_t0         |     0.0436042 |      5.37369 | ↓           |
|      8 | BORE_OIL_VOL_LAG_3_t0    |     0.0376258 |      4.63693 | ↓           |
|      9 | BORE_OIL_VOL_LAG_2_t0    |     0.0359473 |      4.43008 | ↓           |
|     10 | BORE_OIL_VOL_LAG_1_t0    |     0.0342591 |      4.22203 | ↑           |        # ← headers defaults to "keys"

    *Total importance is normalised to 100 %. Direction = monotonic sign of
    the SHAP dependence (Spearman).*  

    **Gini coefficient** of importance distribution: **0.840**