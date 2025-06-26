# Automated SHAP Analysis Report

        **Top Influencers** – Most influential: **BORE_OIL_VOL_LAG_3_t0, BORE_GAS_VOL_t0, BORE_OIL_VOL_LAG_1_t0**. 'BORE_OIL_VOL_LAG_3_t0' ranked #1 in 2 iterations.

        **Model Focus** – Gini went from 0.84 to 0.85 (stable).

        **Learning Stability** – Ranking never stabilised (Spearman < 0.9).

        **Key Drivers Analysis**:
        *   The main features that **positively** drive predictions higher are **'BORE_OIL_VOL_LAG_1_t0' and 'BORE_GAS_VOL_t0'**.
        *   The main features that **negatively** drive predictions lower are **'BORE_OIL_VOL_LAG_3_t0' and 'Prod_Start_Time_t0'**.

        ---
        ## Metrics by Iteration
        |   iteration |     gini | spearman_vs_prev   | top_feature              |
|------------:|---------:|:-------------------|:-------------------------|
|           1 | 0.839699 |                    | BORE_OIL_VOL_LAG_1_t0    |
|         500 | 0.83764  |                    | BORE_OIL_VOL_MEAN_LAG_t0 |
|        1000 | 0.837088 |                    | BORE_OIL_VOL_LAG_1_t0    |
|        1500 | 0.844417 |                    | BORE_GAS_VOL_t0          |
|        2000 | 0.844259 |                    | BORE_OIL_VOL_LAG_3_t0    |
|        2500 | 0.849157 |                    | BORE_OIL_VOL_LAG_3_t0    |

        ---
        *Generated automatically.*