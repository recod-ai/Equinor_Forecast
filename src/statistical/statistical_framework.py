import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
import scipy.stats as sp_stats
import matplotlib.pyplot as plt
import seaborn as sns

from themes import themes


"""
Module Overview:
This module provides a modular framework for the statistical analysis of configuration results.
It includes classes to calculate statistics, rank configurations, perform hypothesis testing,
generate automatic conclusions, and style the resulting DataFrames for display.
"""

# Type alias: each configuration maps to a tuple of (combined_metrics_df, combined_quantile_list)
ResultsDictType = Dict[str, Tuple[pd.DataFrame, List[pd.DataFrame]]]

class StatsCalculator:
    """
    Utility class to calculate basic statistics (mean and std) for a DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def group_statistics(self) -> pd.DataFrame:
        """
        Groups the DataFrame by 'Category' and computes mean and standard deviation for each metric.
        Returns:
            Aggregated DataFrame with statistics.
        """
        stats_df = self.df.groupby("Category").agg({
            "R²": ["mean", "std"],
            "SMAPE": ["mean", "std"],
            "MAE": ["mean", "std"]
        })
        # Flatten MultiIndex columns for a cleaner DataFrame.
        stats_df.columns = ["_".join(col).strip() for col in stats_df.columns.values]
        stats_df.reset_index(inplace=True)
        return stats_df

class RankingStrategy(ABC):
    """
    Abstract base class for ranking strategies.
    """
    @abstractmethod
    def rank(self, metrics_dfs: Dict[str, pd.DataFrame]) -> List[Tuple[str, float]]:
        """
        Ranks configurations based on their metrics.
        """
        pass
    
class CompositeMetricRanking(RankingStrategy):
    """Ranking based on weighted composite of normalized metrics."""
    def __init__(self, weights: Dict[str, float] = {'R²': 0.3, 'SMAPE': 0.5, 'MAE': 0.2}):
        self.weights = weights

    def rank(self, aggregated_metrics: Dict[str, Dict[str, dict]]) -> Dict[str, Dict[str, int]]:
        """
        Ranks configurations within each category.
        
        Args:
            aggregated_metrics: A dict mapping each category (e.g., 'Global', 'Aggregated', 'Cumulative')
                                to another dict, where each key is a configuration and its value is a 
                                dictionary of metric statistics (e.g., {'R²_mean': value, 'SMAPE_mean': value, ...}).
        Returns:
            A dict mapping each category to a dict of configuration rankings.
        """
        rankings = {}
        for category, configs in aggregated_metrics.items():
            scores = {}
            # Normalize each metric across all configurations in the current category.
            for config, stats in configs.items():
                metrics = {
                    'R²': stats['R²_mean'],
                    'SMAPE': stats['SMAPE_mean'],
                    'MAE': stats['MAE_mean']
                }
                normalized = self._normalize_metrics(metrics, [s for s in configs.values()])
                # Compute the weighted composite score.
                composite = sum(self.weights[k] * normalized[k] for k in metrics)
                scores[config] = composite
            # Rank configurations: highest composite score gets rank 1.
            rankings[category] = self._rank_scores(scores)
        return rankings

    def _normalize_metrics(self, metrics: Dict[str, float], all_stats: List[Dict]) -> Dict[str, float]:
        """
        Normalizes each metric across the provided statistics.
        
        For R² (higher is better): normalized = (value - min) / (max - min).
        For SMAPE and MAE (lower is better): normalized = 1 - (value - min) / (max - min).
        """
        # Build lists of values for each metric.
        all_values = {k: [s[f"{k}_mean"] for s in all_stats] for k in metrics}
        return {
            'R²': self._normalize(metrics['R²'], all_values['R²'], inverse=False),
            'SMAPE': self._normalize(metrics['SMAPE'], all_values['SMAPE'], inverse=True),
            'MAE': self._normalize(metrics['MAE'], all_values['MAE'], inverse=True)
        }

    @staticmethod
    def _normalize(value: float, values: list, inverse: bool) -> float:
        """
        Scales a value between 0 and 1 based on min and max of the provided list.
        
        If inverse is True (for metrics where lower is better), the normalized value is inverted.
        """
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return 0.5  # All values are the same.
        scaled = (value - min_val) / (max_val - min_val)
        return (1 - scaled) if inverse else scaled

    @staticmethod
    def _rank_scores(scores: Dict[str, float]) -> Dict[str, int]:
        """
        Ranks the configurations based on their composite score.
        The highest score receives rank 1.
        """
        sorted_configs = sorted(scores, key=scores.get, reverse=True)
        return {config: rank for rank, config in enumerate(sorted_configs, start=1)}


class ConfigAnalysis:
    """
    Analyzes a single configuration's results.
    """
    def __init__(self, config_key: str, metrics_df: pd.DataFrame, quantile_list: List[pd.DataFrame]):
        self.config_key = config_key
        self.metrics_df = metrics_df
        self.quantile_list = quantile_list

    def analyze_metrics(self) -> pd.DataFrame:
        """
        Computes grouped statistics for the combined metrics.
        Returns:
            A DataFrame with aggregated statistics.
        """
        return StatsCalculator(self.metrics_df).group_statistics()

    def group_statistics_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Groups the quantile DataFrame by its base category (Global, Aggregated, or Cumulative)
        and computes statistics.
        """
        df = df.copy()
        df["BaseCategory"] = df["Category"].apply(lambda x: x.split()[0])
        stats_df = df.groupby("BaseCategory").agg({
            "R²": ["mean", "std"],
            "SMAPE": ["mean", "std"],
            "MAE": ["mean", "std"]
        })
        stats_df.columns = ["_".join(col).strip() for col in stats_df.columns.values]
        stats_df.reset_index(inplace=True)
        return stats_df

    def analyze_quantiles(self) -> List[pd.DataFrame]:
        """
        Computes statistics for each quantile DataFrame.
        Returns:
            A list of DataFrames with aggregated quantile statistics.
        """
        return [self.group_statistics_quantiles(q_df) for q_df in self.quantile_list]

    def hypothesis_test(self, metric: str, group1: str, group2: str) -> Tuple[float, float]:
        """
        Performs a t-test comparing the specified metric between two groups.
        Returns:
            Tuple of (t_statistic, p_value).
        """
        data1 = self.metrics_df[self.metrics_df["Category"] == group1][metric]
        data2 = self.metrics_df[self.metrics_df["Category"] == group2][metric]
        return sp_stats.ttest_ind(data1, data2, equal_var=False, nan_policy="omit")

class StatisticalFramework:
    """
    Main framework class to manage the statistical analysis process.
    """
    def __init__(self, results_dict: ResultsDictType):
        self.results_dict = results_dict
        # Create a ConfigAnalysis instance for each configuration.
        self.config_analyses = {
            config_key: ConfigAnalysis(config_key, metrics_df, quantile_list)
            for config_key, (metrics_df, quantile_list) in results_dict.items()
        }

    def run_analysis(self) -> Dict[str, Dict[str, object]]:
        """
        Runs the analysis for each configuration.
        Returns:
            Dictionary mapping configuration keys to their metrics and quantile statistics.
        """
        analysis_results = {}
        for config_key, analysis in self.config_analyses.items():
            analysis_results[config_key] = {
                "metrics_stats": analysis.analyze_metrics(),
                "quantiles_stats": analysis.analyze_quantiles()
            }
        return analysis_results

    def get_aggregated_metrics(self) -> Dict[str, Dict[str, dict]]:
        """
        Aggregates metrics statistics from each configuration by category.
        Returns:
            A dictionary mapping each category (e.g. 'Global', 'Aggregated', 'Cumulative')
            to a dictionary mapping each configuration to its metrics statistics.
        """
        aggregated = {}
        for config, analysis in self.config_analyses.items():
            stats_df = analysis.analyze_metrics()
            for _, row in stats_df.iterrows():
                category = row['Category']
                if category not in aggregated:
                    aggregated[category] = {}
                aggregated[category][config] = {
                    'R²_mean': row['R²_mean'],
                    'SMAPE_mean': row['SMAPE_mean'],
                    'MAE_mean': row['MAE_mean']
                }
        return aggregated

    def perform_hypothesis_tests(self, metric: str, group1: str, group2: str) -> Dict[str, Dict[str, float]]:
        """
        Performs t-tests for each configuration comparing a given metric between two groups.
        Returns:
            Dictionary mapping configuration keys to t-test results.
        """
        test_results = {}
        for config, analysis in self.config_analyses.items():
            t_stat, p_val = analysis.hypothesis_test(metric, group1, group2)
            test_results[config] = {"t_stat": t_stat, "p_value": p_val}
        return test_results

class HypothesisTester:
    """
    Provides methods to generate automatic conclusions from hypothesis test results.
    """
    @staticmethod
    def generate_conclusions(test_results: dict, metric: str, group1: str, group2: str,
                             significance_level: float = 0.05) -> dict:
        """
        Generates conclusions for hypothesis tests based on p-values and t-statistics.
        Args:
            test_results: Dictionary with test results per configuration.
            metric: The metric tested (e.g., "R²", "SMAPE").
            group1: Name of the first group.
            group2: Name of the second group.
            significance_level: Significance level (default 0.05).
        Returns:
            Dictionary with conclusions for each configuration.
        """
        conclusions = {}
        for config, result in test_results.items():
            t_stat = result['t_stat']
            p_value = result['p_value']
            if p_value < significance_level:
                if t_stat > 0:
                    conclusion = (f"For configuration '{config}', data indicates that group '{group1}' has, on average, higher "
                                  f"{metric} than group '{group2}' (t={t_stat:.3f}, p={p_value:.3f}).")
                else:
                    conclusion = (f"For configuration '{config}', data indicates that group '{group2}' has, on average, higher "
                                  f"{metric} than group '{group1}' (t={t_stat:.3f}, p={p_value:.3f}).")
            else:
                conclusion = (f"For configuration '{config}', there is no statistically significant difference in {metric} "
                              f"between groups '{group1}' and '{group2}' (t={t_stat:.3f}, p={p_value:.3f}).")
            conclusions[config] = conclusion
        return conclusions

class DataFrameStyler:
    """
    Applies a custom style to DataFrames based on a selected theme.
    """
    @staticmethod
    def style_dataframe(df: pd.DataFrame, theme_name: str, themes: dict):
        """
        Styles a DataFrame using the chosen theme.
        Args:
            df: The DataFrame to style.
            theme_name: Name of the theme.
            themes: Dictionary of available themes.
        Returns:
            A styled DataFrame.
        """
        chosen_theme = themes.get(theme_name, themes["minimal"])
        styled_df = (
            df.style
              .format({
                  "R²_mean": "{:.2f}",
                  "R²_std": "{:.2f}%",
                  "SMAPE_mean": "{:.2f}",
                  "SMAPE_std": "{:.2f}",
                  "MAE_mean": "{:.2f}%",
                  "MAE_std": "{:.2f}"
              })
              .set_properties(**{
                  'font-size': '12pt',
                  'text-align': 'center',
                  'color': chosen_theme["text"],
                  'background-color': chosen_theme["bg"]
              })
              .background_gradient(cmap='coolwarm', subset=['R²_mean', 'SMAPE_mean', 'MAE_mean'])
              .set_table_styles([
                  {
                      'selector': 'th',
                      'props': [
                          ('background-color', chosen_theme["accent"]),
                          ('color', chosen_theme["bg"]),
                          ('font-size', '14pt'),
                          ('text-align', 'center'),
                          ('border', f'1px solid {chosen_theme["grid"]}')
                      ]
                  },
                  {
                      'selector': 'td',
                      'props': [
                          ('font-size', '12pt'),
                          ('text-align', 'center'),
                          ('border', f'1px solid {chosen_theme["grid"]}')
                      ]
                  },
                  {
                      'selector': 'table',
                      'props': [
                          ('background-color', chosen_theme["bg"]),
                          ('border-collapse', 'collapse')
                      ]
                  }
              ])
        )
        return styled_df
    
    

class DataFrameStylerAuto:
    """
    Automatically styles a DataFrame based on its structure and the selected theme.
    """
    @staticmethod
    def style_dataframe(df: pd.DataFrame, theme_name: str, themes: dict):
        """
        Inspect the DataFrame to detect numeric and statistical columns, apply formatting,
        background gradients, and theme-based styling.

        Args:
            df: The DataFrame to style.
            theme_name: Name of the theme to use.
            themes: A dict mapping theme names to theme property dicts.

        Returns:
            A pandas Styler object with the applied theme.
        """
        # Retrieve chosen theme, fallback to 'minimal'
        chosen_theme = themes.get(theme_name, themes.get("minimal", {}))

        # Print basic DataFrame info (shape and index range)
        try:
            idx_min = df.index.min()
            idx_max = df.index.max()
        except Exception:
            idx_min = None
            idx_max = None
        print(f"DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns; "
              f"Index range: {idx_min} to {idx_max}")

        # Detect numeric columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # Build formatting dict and identify gradient columns
        fmt: dict = {}
        gradient_cols: list = []
        for col in numeric_cols:
            lower = col.lower()
            if lower.endswith("_mean"):
                fmt[col] = "{:.2f}"
                gradient_cols.append(col)
            elif lower.endswith("_std"):
                fmt[col] = "{:.2f}%"
            else:
                fmt[col] = "{:.2f}"

        # Begin styling
        styler = df.style.format(fmt)

        # Apply general cell properties
        styler = styler.set_properties(**{
            'font-size': chosen_theme.get('font_size', '12pt'),
            'text-align': 'center',
            'color': chosen_theme.get('text', '#000'),
            'background-color': chosen_theme.get('bg', '#fff')
        })

        # Apply background gradient if applicable
        if gradient_cols:
            styler = styler.background_gradient(
                subset=gradient_cols,
                cmap=chosen_theme.get('gradient_cmap', 'coolwarm')
            )

        # Define table styles
        table_styles = [
            {
                'selector': 'th',
                'props': [
                    ('background-color', chosen_theme.get('accent', '#333')),
                    ('color', chosen_theme.get('bg', '#fff')),
                    ('font-size', chosen_theme.get('header_font_size', '14pt')),
                    ('text-align', 'center'),
                    ('border', f"1px solid {chosen_theme.get('grid', '#ccc')}")
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('border', f"1px solid {chosen_theme.get('grid', '#ccc')}")
                ]
            },
            {
                'selector': 'table',
                'props': [
                    ('border-collapse', 'collapse'),
                    ('background-color', chosen_theme.get('bg', '#fff'))
                ]
            }
        ]
        styler = styler.set_table_styles(table_styles)

        return styler
