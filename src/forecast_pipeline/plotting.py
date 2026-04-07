# forecast_pipeline/plotting.py
"""
forecast_pipeline.plotting
--------------------------
Unified visualization layer for the forecast pipeline.
This module acts as a facade, re-exporting functions from internal
plotting modules to maintain a stable public API.
"""
# --- Local Application Imports: Facade Exports ---
from ._plotting_core import (
    _plot_integrated_view_from_agg,
    _plot_seq,
    plot_integrated_view,
    plot_predictions_wrapper,
    plot_series,
    plot_darts_integrated,
    plot_arps_integrated_from_point,
    plot_by_well_advanced,
)
from ._plotting_reports import (
    plot_campaign_strategy_performance,
    plot_champions_well,
    plot_hyperparameter_importance_per_well,
    plot_performance_by_architecture,
    render_champions_view_auto,
)

__all__ = [
    # Core series plotting
    "_plot_seq",
    "_plot_integrated_view_from_agg",
    "plot_series",
    "plot_predictions_wrapper",
    "plot_integrated_view",
    "plot_darts_integrated",
    "plot_arps_integrated_from_point",
    "plot_by_well_advanced",
    # Reporting and diagnostics
    "render_champions_view_auto",
    "plot_performance_by_architecture",
    "plot_champions_well",
    "plot_campaign_strategy_performance",
    "plot_hyperparameter_importance_per_well",
]