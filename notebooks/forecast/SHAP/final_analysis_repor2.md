# Automated SHAP Analysis Report

        **Top Influencers** – Most influential: **BORE_OIL_VOL_LAG_1_t0, Prod_Start_Time_t0, BORE_OIL_VOL_LAG_4_t0**. 'BORE_OIL_VOL_LAG_1_t0' ranked #1 in 4 iterations.

        **Model Focus** – Gini went from 0.84 to 0.83 (stable).

        **Learning Stability** – Ranking never stabilised (Spearman < 0.9).

        **Key Drivers Analysis**:
        *   The main features that **positively** drive predictions higher are **'Prod_Start_Time_t0' and 'BORE_OIL_VOL_LAG_2_t0'**.
        *   The main features that **negatively** drive predictions lower are **'BORE_OIL_VOL_LAG_1_t0' and 'BORE_OIL_VOL_LAG_4_t0'**.

        ---
        ## Metrics by Iteration
        |   iteration |     gini | spearman_vs_prev   | top_feature           |
|------------:|---------:|:-------------------|:----------------------|
|           1 | 0.840842 |                    | Prod_Start_Time_t0    |
|         500 | 0.836672 |                    | BORE_OIL_VOL_LAG_1_t0 |
|        1000 | 0.842776 |                    | BORE_OIL_VOL_LAG_1_t0 |
|        1500 | 0.836701 |                    | BORE_OIL_VOL_LAG_1_t0 |
|        2000 | 0.837941 |                    | BORE_OIL_VOL_LAG_6_t0 |
|        2500 | 0.834711 |                    | BORE_OIL_VOL_LAG_1_t0 |

        ---
        *Generated automatically.*