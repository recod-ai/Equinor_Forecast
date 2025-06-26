# SHAP Report – Iteration 2000

    **Top 10 features**

    | Rank | Feature | Mean |SHAP| | Share (%) | Direction |
    |------|---------|-------------|-----------|-----------|
    |   rank | feature                  |   mean_|SHAP| |   relative_% | direction   |
|-------:|:-------------------------|--------------:|-------------:|:------------|
|      1 | BORE_OIL_VOL_LAG_6_t0    |     0.16337   |     20.2175  | ↓           |
|      2 | BORE_GAS_VOL_t0          |     0.156323  |     19.3455  | ↓           |
|      3 | BORE_OIL_VOL_LAG_7_t0    |     0.102977  |     12.7437  | ↑           |
|      4 | BORE_OIL_VOL_LAG_1_t0    |     0.0774609 |      9.58602 | ↑           |
|      5 | Prod_Start_Time_t0       |     0.0732292 |      9.06234 | ↑           |
|      6 | BORE_OIL_VOL_LAG_5_t0    |     0.0481751 |      5.96181 | ↓           |
|      7 | BORE_OIL_VOL_MEAN_LAG_t0 |     0.0467241 |      5.78225 | ↑           |
|      8 | ON_STREAM_HRS_t0         |     0.0386496 |      4.78301 | ↑           |
|      9 | BORE_OIL_VOL_LAG_2_t0    |     0.0360336 |      4.45927 | ↓           |
|     10 | BORE_OIL_VOL_LAG_3_t0    |     0.035481  |      4.39088 | ↓           |        # ← headers defaults to "keys"

    *Total importance is normalised to 100 %. Direction = monotonic sign of
    the SHAP dependence (Spearman).*  

    **Gini coefficient** of importance distribution: **0.840**