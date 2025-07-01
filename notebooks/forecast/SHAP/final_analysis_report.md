# Automated SHAP Analysis Report

        **Top Influencers** — Most influential: **BORE_OIL_VOL_MEAN_LAG_t0, BORE_GAS_VOL_t0, BORE_OIL_VOL_LAG_6_t0**. 'BORE_OIL_VOL_MEAN_LAG_t0' ranked #1 in 1 iterations.

        **Model Focus** — Gini went from 0.84 to 0.84 (stable).

        **Learning Stability** — Ranking never stabilized (Spearman < 0.9).

        **Key Drivers Analysis**:
        * Positive drivers: **BORE_GAS_VOL_t0** and **BORE_OIL_VOL_MEAN_LAG_t0**.
        * Negative drivers: **Prod_Start_Time_t0** and **BORE_OIL_VOL_LAG_1_t0**.

        ---
        ## Metrics by Iteration
        |   iteration |     gini | spearman_vs_prev   | top_feature              |
|------------:|---------:|:-------------------|:-------------------------|
|           1 | 0.837632 |                    | Prod_Start_Time_t0       |
|         500 | 0.835554 |                    | BORE_OIL_VOL_MEAN_LAG_t0 |
|        1000 | 0.839568 |                    | BORE_GAS_VOL_t0          |
|        1500 | 0.836715 |                    | BORE_GAS_VOL_t0          |
|        2000 | 0.839896 |                    | BORE_OIL_VOL_LAG_6_t0    |
|        2500 | 0.83977  |                    | BORE_OIL_VOL_LAG_5_t0    |

        ---
        *Generated automatically.*