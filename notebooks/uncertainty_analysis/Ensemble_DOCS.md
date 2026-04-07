# HPO Ensemble & Evaluation Pipeline — Technical Documentation

> **Scope**: This document explains the architecture, data model, algorithms, and orchestration of the post‑hoc analysis pipeline you shared ("Phase 4"). It is written to be useful both for specialists maintaining the code and for advanced users integrating it in larger systems.

---

## 1) Executive Overview

The system turns heterogeneous *forecast experiment outputs* (per‑job JSON results) into *aligned, queryable time series* and *publication‑quality evaluation artifacts*. It consolidates predictions across architectures (e.g., **Seq2PINN** and **ARPS**) into:

* **Intra‑family ensembles** (average within a modeling family)
* **Inter‑family final ensemble** (average of family means)
* **Uncertainty bands** with **validation‑tuned Gaussian scaling**
* **Metrics** (SMAPE, coverage, sharpness) computed exactly as plotted
* **Risk CDFs** of accumulated production over flexible horizons

All downstream steps operate on a **Series Store** (Parquet) and a small **metadata layer** (boundaries & full history), enabling reproducibility and quick re‑analysis without re‑running training.

---

## 2) End‑to‑End Data Flow

```mermaid
flowchart TD
  subgraph Execution
    J[Per-job JSON results]
  end

  subgraph Series Store
    SS[(Parquet series)]
    META[meta/boundaries.parquet\n history/well=.../history.parquet]
  end

  subgraph Phase 4 Pipeline
    S1[Stage 1\nChampion Harvest]
    S2[Stage 2\nSeries Loader]
    S25[Stage 2.5\nRestrict Train]
    S3[Stage 3\nIntra & Inter Ensembles]
    S4[Stage 4\nVisualization]
    S41[Stage 4.1\nRisk (in‑memory)]
  end

  J --> |normalize→record| SS
  J --> |context→meta| META
  SS --> S2
  META --> S2
  S1 --> S2
  S2 --> S25 --> S3 --> S4
  S3 --> S41
```

**Key invariants**

* All plots, tables, and risk numbers use the same **axis alignment** policy: `global_idx → t` using split‑dependent offsets.
* Coverage metrics and fan bands share the same **band construction** logic.
* When `ci_mode='quantile'`, empirical quantiles are used **as‑is** (no rescale).

---

## 3) Storage Model (Series Store)

**Path layout (Hive‑like partitions)**

```
<base_dir>/group=<G>/arch=<A>[/dataset=<D>][/well=<W>]/campaign=<C>/job=<HASH>.parquet
```

**Metadata**

* `meta/boundaries.parquet` — authoritative per‑well split indices: `train_end`, `val_end`, `test_start` (+ optional `group`, `dataset`).
* `history/well=<WELL>/history.parquet` — full ground truth `ytrue` with axis `t`.

**Record builder (ingestion)**

* `build_series_record(result_dict)` consolidates multiple legacy/new JSON shapes into a robust `record` with:

  * `series_rows`: rows of `(split, t, idx, yhat, [ytrue])` for `val/test` and (optionally) `train` if present under `series_context.forecast_agg.train`.
  * `context.boundaries` and `context.full_history` when available.
  * `meta`: `{group, arch, campaign, job_hash, dataset, well, H, L}`.

**Persistence**

* `persist_series(record, settings)` → persist_series(record, settings): Writes job files using an atomic rename strategy (write to temp → os.replace), preventing partial reads if the process crashes mid-write.
* `persist_boundaries_manifest(record, settings)` → Upserts metadata protected by a file lock (.lock via fcntl), ensuring safe concurrent access by multiple workers.
* `persist_full_history(record, settings)` → Atomically writes/extends history, also protected by file locks per well.

> ⚠️ **Concurrency note**: current writes are idempotent but not atomic/locked. If true multi‑process safety is needed, add temp‑file writes + `os.replace()` and a manifest lock.

---

## 4) Orchestrator (Phase 4)

**Module**: `common/phase_orchestrator.py`

### 4.1 Configuration (`Phase4Config`)

* **Champion selection**: `scoring_strategy` in `{"weighted_score","robust_score"}`; `metric_weights`; `lower_is_better` flags (defaults set `weighted_score=True`, `robust_score=True`).
* **Campaigns**: `campaigns_to_ensemble` maps split tags to experiment groups.
* **Top‑K**: `n_champions_per_group` and optional post‑hoc overrides.
* **Visualization**: palette & toggles for family/champion traces.
* **Logging**: `log_mode` = `compact|verbose` (with compact block renderers).
* **Risk**: split selectors, horizons, weighting strategy, temperature.
* **Paths**: `project_root`, `series_store_root` (auto‑resolved when `None`).

### 4.2 Runtime Dependency Loader

`_import_runtime_deps()` lazily imports only when needed (keeps module import side‑effects minimal):

* Champion selection/scoring (`hpo.analysis_utils`)
* Series reader (`common.series_store_reader.read_series_by_champions`)
* Ensemble builders (`hpo.ensemble_ops`)
* Plotting (`plotting.ensemble_plots.plot_conjugated_ensemble`)
* Viz helpers (`common.phase_viz_support`)

### 4.3 Pipeline Stages

**Stage 1 — Champion Harvest** (`_stage1_select_champions`)

* Reads campaign leaderboards, applies selected scoring (`weighted_score` or `robust_score`).
* Normalizes `arch` using `_ARCH_PARTITION_MAP` (e.g., `Seq2PINN→seq2`, `Arps→arps`).
* Optional **per-(well,arch) Top‑K** clamp via `_enforce_topk_per_well_arch` (off by default).
* Produces `champions_df` + the effective `score_col` + `posthoc` used.

**Stage 2 — Series Loader** (`_stage2_load_series`)

* Calls `read_series_by_champions` against the Series Store for `champions_df`.
* Returns `series_df` (row‑wise predictions), `boundaries_df` (manifest), `full_history_by_well` (map of `t,ytrue`).

**Stage 2.5 — Train Restriction** (`_stage25_restrict_train`)

* Computes **one primary job per well** via `_compute_primary_per_well` on the chosen score (lower is better).
* `_restrict_train_to_primary` keeps only primary rows for the `train` split; `val/test` are untouched.
* Adds `is_primary` convenience flag.

**Stage 3 — Ensembles** (`_stage3_build_ensembles`)

* Intra‑family ensemble: family means per `(well, arch, split, t)`.
* Inter‑family final ensemble: average of family means across `arch` for each well.
* Attaches **member counts**:

  * `_n_members_by_family(champions_df)` → merge into `intra_family_df`.
  * `_n_members_by_well_total(...)` → merge into `final_ensemble_df`.

**Stage 4 — Visualization & Reporting** (`_stage4_visualize_and_report`)

* Picks wells to plot (config list or inferred from data).
* **Axis alignment**: builds `align_with_t_factory(fh,bounds)` → converts each frame’s `(idx,split)` into `global_idx` using offsets and merges to `t`.
* Plots:

  1. Intra Seq2 (if available)
  2. Intra ARPS (if available)
  3. Inter final ensemble (always)
  4. **Members per family** (spaghetti) with bold **family mean** when present; otherwise fallback to inter‑family mean.

**Stage 4.1 — Risk (compute‑only)** (`_stage41_compute_risk_curves`)

* For each `(well, arch)` and for each `(split selector, horizon)`:

  1. Build aligned per‑member series (`risk_core.build_member_series`).
  2. Compute member weights (`uniform` or `distance_softmax(temp)`).
  3. Accumulate production within the window (`accumulate_window`).
  4. Compute weighted quantiles P10/P50/P90 and mean/min/max.
* Returns an **in‑memory** dict `{well: DataFrame}`. No disk I/O.

---

## 5) Axis Alignment & Split Offsets

**Factory**: `align_with_t_factory(full_history_df, bounds)` returns a function that:

1. Detects and drops fake `t==idx` placeholders (numeric) to avoid double counting.
2. Builds `global_idx` from `idx` using per‑split offsets:

   * `VAL_OFFSET = train_end + 1`
   * `TEST_OFFSET = max(val_end, test_start) + 1`
3. Inner‑joins `global_idx` to a `t_map` built from `full_history_df.index → t`.

This guarantees that **series**, **intra**, and **inter** data land on the **same physical axis**.

---

## 6) Metrics & Uncertainty (Plot/Stats Parity)

**Module**: `plotting/ensemble_stats.py`

* **SMAPE**: `_smape(y_true, y_pred, eps=1e-9)` implements the canonical formula with an `eps` stabilizer in the denominator. When both values are near zero, the contribution is effectively 0.

* **Coverage & Sharpness**: `_compute_coverage_and_sharpness(g, ci_mode, coverage, ...)` builds `(lower,upper)` bands and returns:

  * Coverage: fraction of `ytrue` within `[lower, upper]`.
  * Sharpness: median width `median(upper - lower)`.

* **Band construction**: `_compute_bounds_for_mode(g, ci_mode, ci_k, ci_quantiles, ...)`:

  * `ci_mode='std'`: `mean ± k·std`
  * `ci_mode='sem'`: `mean ± k·(std/√n_members)`
  * `ci_mode='quantile'`: uses provided `qlow/qhigh` columns; **no rescale**

* **Validation‑tuned scaling**: `calibrate_k_on_validation(df_final_vt, ...)` grid‑searches an effective scalar `k_eff` so empirical coverage on **Validation** is as close as possible to the target (tie‑break on smaller sharpness). Applies to `'std'|'sem'`; ignored for `'quantile'`.

> Default z‑scores are fetched via `_z_for_coverage` (e.g., 90%→1.645). You can override with `override_k`.

---

## 7) Plotting Layer

### 7.1 Conjugated Ensemble Plot

**Module**: `plotting/ensemble_plots.py` (assumed from usage)

* Overlays **final ensemble (inter)** and **family means (intra)** on one canvas.
* Regions shaded by split using boundaries.
* Optional champion traces and train reconstruction toggles.
* Shares metric computation with tables to ensure parity.

### 7.2 Members Plot (Per Family)

**Module**: `plotting/ensemble_members.py`

* **Purpose**: spaghetti of per‑member `yhat` for VAL/TEST, plus **family mean** when available; fallback to **final mean** otherwise.
* **Input preparation**: `build_members_frame_for_well(series_df, well, ...)` returns per‑member lines with normalized `split` and “best‑effort” canonical columns.
* **Realignment**: enforces `global_idx→t` even if someone left `t==idx` placeholders.
* **Legends**: de‑duplicated by split (`Member (Val)` vs `Member (Test)`).
* **Caps**: `max_members` limits plotted traces for responsiveness.

### 7.3 Risk CDF Plot

**Module**: `plotting/risk_plots.py`

* Pastel, white‑canvas **Empirical CDF**:

  * Filled area under curve
  * P10–P90 translucent band
  * Rug markers (density hint)
  * Quantile markers with tooltips
  * Side “stats card” (n, mean, P50, P10, P90, range)
* API:

  * `plot_risk_cdf(accum_values, weights=None, title=..., palette=..., ...)`
  * `plot_risk_cdf_for(series_df, final_ensemble_df, boundaries_df, full_history_by_well, well, arch, selector, horizon_days, weighting, temp, ...)` orchestrates members→weights→accumulate→plot for a single well.

> **Theme**: `_build_pastel_theme` + `_apply_clean_white_canvas` enforce a light aesthetic and readable axes. No dark backgrounds.

---

## 8) Risk Core (Conceptual)

While the *core* implementations are referenced from `risk.risk_core`, the orchestration makes the following guarantees:

* **Member series** are re‑aligned to `t` using the same factory and boundaries used by Stage 4 views.
* **Weighting** strategies:

  * `uniform`: equal weights
  * `distance_softmax(temp)`: more weight to members closer to the family mean (on Validation); temperature controls sharpness
* **Windows** are defined by `selector ∈ {train,val,test,val+test}` and `horizon_days` starting at that window’s beginning (`-1` means until its end).
* **Outputs** per `(well, arch, selector, horizon)` include weighted `q10/q50/q90`, `mean`, `min`, `max`, and `n_members` used.

---

## 9) Utility Layer (selected)

### 9.1 `common/phase_viz_support.py`

* `build_full_history(series_df, well, full_history_by_well=...)` → chooses authoritative history, else warns.
* `infer_boundaries(final_ensemble_df, series_df, well, boundaries_df=None, full_history_df=None)` → rules:

  1. if manifest row exists → use it
  2. else infer from available splits: `max idx` for train/val, `min idx` for test
  3. enforce monotonicity and clamp to history length
* `lookup_manifest_bounds(boundaries_df, well)` and minimal fallbacks for history/bounds.
* `align_with_t_factory(full_history_df, bounds)` → **the** alignment engine used everywhere.
* `filter_intra_by_arch(df, 'seq2'|'arps')`, `debug_frame(...)`, `make_family_traces_visible(fig)` convenience helpers.

### 9.2 Train Restrict Helpers

* `_compute_primary_per_well(champions_df, score_col='weighted_score')` → one primary `job_hash` per well (lower score = better; falls back to `val_smape_agg`).
* `_restrict_train_to_primary(series_df, primary_by_well)` → keep only primary rows inside `train`; mark `is_primary`.

---

## 10) API Contracts (selected)

### 10.1 Calibration

```python
calibrate_k_on_validation(
    df_final_vt, full_history_df,
    ci_mode='sem', target_coverage=0.90,
    mean_col='yhat_final_mean', std_col='std_final', n_members_col='n_members',
    qlow_col='yhat_q_low_final', qhigh_col='yhat_q_high_final',
    k_grid=None,
) -> {
  'k_eff': float,
  'cov_val': float, 'sharp_val': float, 'n_points': int
}
```

* Applies only to `std|sem`. For `quantile`, returns `k_eff=1.0` and the empirical coverage.

### 10.2 Coverage/Sharpness

```python
_compute_coverage_and_sharpness(g, ci_mode, coverage, mean_col, std_col, n_members_col,
                                qlow_col=None, qhigh_col=None, override_k=None) -> (cov, sharp)
```

* `override_k` lets you inject a pre‑calibrated multiplier (e.g., `z*keff`).

### 10.3 Members Frame

```python
build_members_frame_for_well(series_df, well, member_id_col='job_hash', yhat_col='yhat') -> DataFrame
```

* Returns columns subset `['well','split','idx|global_idx','t','arch',member_id_col,yhat_col]` where available; normalizes `split`.

### 10.4 Risk Orchestration

```python
plot_risk_cdf_for(..., well, arch, selector, horizon_days,
                  weighting='uniform', temp=0.5, palette='default', ...) -> go.Figure
```

* Uses the same alignment and boundary rules as plots.

---

## 11) Defaults & Tunables

* **Coverage levels**: defaults use `_z_for_coverage(coverage)` (e.g., 90%→1.645). Fan levels in code default to **50% & 90%**; extend as desired.
* **CI mode**: `'sem'` is common for ensembles (tighter bands with more members).
* **Top‑K champions**: off by default; enable via `FORCE_STAGE1_TOPK` and `_enforce_topk_per_well_arch`.
* **Risk horizons**: e.g., `[300, 600, 900, -1]` where `-1` means “until end of window”.
* **Weighting**: `uniform` or `distance_softmax` with `temp`.

---

## 12) Edge Cases & Safeguards

* **Missing columns**: builders and plotters perform presence checks and fall back (e.g., no `std_final` → degenerate band at mean; no `quantiles` → fall back to `std`).
* **Unknown `ci_mode`**: coerced to `'std'`.
* **No history/boundaries**: fallbacks create minimal viable views.
* **`t == idx` numeric placeholders**: dropped before realignment to avoid mis‑axis.
* **Empty groups**: stages log and skip gracefully.
* **Members cap**: protects rendering performance.

---

## 13) How to Run

```python
from common.phase_orchestrator import run_phase4_pipeline, Phase4Config

cfg = Phase4Config(
    scoring_strategy='weighted_score',
    wells_to_analyze=['P11','P12'],
    log_mode='compact',
    enable_risk_plots=True,
)

artifacts = run_phase4_pipeline(cfg)
final_df = artifacts['final_ensemble_df']
risk_by_well = artifacts['risk_outputs_by_well']
```

---

## 14) Known Limitations & Future Work

* **Atomicity/locking**: add temp‑file writes + `os.replace()` and a small lock for manifest updates if you expect concurrent writers.
* **Formal conformal prediction**: current calibration scales Gaussian bands; a true conformal method could be added as an option.
* **Fan gradients**: extend fan defaults (e.g., 50/70/90/95) and a color ramp if desired.
* **Dataset‑aware manifest**: if multiple datasets share wells, ensure manifest queries always constrain on `(well, group, dataset)` (code already attempts this in `persist_boundaries_manifest`).

---

## 15) Glossary

* **Intra‑family**: aggregation within a modeling family (e.g., all Seq2PINN).
* **Inter‑family (final)**: aggregation across families (meta‑ensemble of means).
* **Coverage**: fraction of `ytrue` contained by the predicted interval.
* **Sharpness**: width of the interval (narrower is better, all else equal).
* **SEM**: standard error of the mean: `std/√n_members`.
* **Global index**: monotone timeline index combining train/val/test via offsets.

---

## 16) File/Module Map (selected)

```
src/
├── common/
│   ├── phase_orchestrator.py      # main controller for Phase 4
│   ├── phase_viz_support.py       # alignment/boundaries/history helpers
│   ├── train_restrict.py          # primary-per-well + train filtering
│   └── ...
├── plotting/
│   ├── ensemble_stats.py          # SMAPE, coverage, sharpness, calibration
│   ├── ensemble_plots.py          # conjugated inter+intra plot (used by Stage 4)
│   ├── ensemble_members.py        # spaghetti members + family mean
│   └── risk_plots.py              # pastel risk CDF plots
└── risk/
    └── risk_core.py               # member building, weights, accumulate, quantiles
```

---

### Appendix A — Pseudocode for Coverage Calibration

```text
for k in linspace(0.25, 3.50):
  lower, upper = build_bounds(ci_mode, k*z(target_coverage))
  cov  = mean(ytrue in [lower, upper])
  sharp= median(upper - lower)
  err  = |cov - target_coverage|
  keep best by (err ↑) then (sharp ↓)
return k_eff
```

### Appendix B — Split Offsets Example

```
VAL_OFFSET  = train_end + 1
TEST_OFFSET = max(val_end, test_start) + 1

# global_idx for a row r
if r.split in {"val","validation"}:  gi = r.idx + VAL_OFFSET
elif r.split == "test":                 gi = r.idx + TEST_OFFSET
else:                                     gi = r.idx
```
