# HPO Pipeline — Practical Overview

This repository implements an end-to-end **Hyperparameter Optimization (HPO)** workflow for forecasting models, plus a **post-hoc analysis layer** that filters results and selects **per-well champions** in a deterministic and auditable way.

The system is designed to answer two practical questions:

1. **Which trials are “good enough” and not suspicious/outliers?**  
2. **Which model(s) should I deploy per well / architecture / strategy — without leaking TEST metrics into the decision?**

---

## What the program can do

### 1) Run HPO campaigns (Optuna or static profiles)
- Generate trial configurations from a defined **search space** (per architecture family).
- Execute the training/forecasting pipeline for each trial.
- Collect results into **leaderboards** (typically CSVs per cycle/run).
- Support iterative HPO loops (multiple cycles) or straight “profile runs”.

### 2) Build a master leaderboard
- Load multiple leaderboard.csv files (supports nested and legacy layouts).
- Aggregate into one **master dataframe** for downstream analysis.
- (Optionally) tolerate empty/corrupt CSVs and still proceed with what is valid.

### 3) Post-hoc filtering (VAL-only) + champion selection (VAL-only)
- Enforce **anti-leak**: selection never sees `test_*` columns.
- Apply robust filtering gates on validation metrics:
  - Quantile thresholds (per well)
  - MAD outlier guards (per well)
  - Optional validation cumulative “band gate” (`val_smape_cum`) with adaptive logic
  - Optional Pareto marking (annotation-only)
- From the surviving pool, select **final champions per well**, using deterministic sorting and tie-breaks.

### 4) Validation ecosystem (retrain & compare)
- Take champions from HPO, generate validation profiles, run validation campaigns.
- Merge HPO results with validation outputs and produce equivalence/delta reports.
- Compute audit metrics after the fact (including TEST-based diagnostics), without affecting selection.

### 5) Reporting and canonicalization
- Normalize results from different families (Seq2 / Arps / Darts) into a consistent schema.
- Produce consolidated top-N views per well and architecture-level summary tables.

---

## Mental model: how modules communicate

The pipeline is easiest to understand as a sequence of layers:

### A) Definition layer (source of truth)
**Purpose:** decide *what* metric matters and *how* it should be interpreted.

- `posthoc_config.py`  
  Holds configuration: metrics, gates, selection knobs, column names, etc.

- `score_semantics.py`  
  Ensures columns exist / are aliased (e.g., distinguishing robust score variants).

- `score_ordering.py`  
  Decides whether a metric is **lower-is-better** or **higher-is-better**.

These three modules establish the rules before any filtering/selection begins.

---

### B) Execution layer (generate and run trials)
**Purpose:** run experiments and produce leaderboards.

- `search_space.py` → defines hyperparameter spaces per architecture family  
- `optuna_utils.py` → creates studies, generates trials, reports results  
- `profile_generators.py` → creates static grid profiles (no Optuna)  
- `hpo_runner.py` → executes the campaign loop and writes results

Outputs from this layer are the leaderboards (CSV) and artifacts used later.

---

### C) Post-hoc analysis layer (filter & select champions)
**Purpose:** turn raw trial results into deployable picks, safely and deterministically.

- `posthoc_filtering.py` (Legacy orchestrator)
  - Loads & sanitizes data
  - Drops `test_*` columns (anti-leak hard rule)
  - Computes thresholds (quantiles + MAD)
  - Applies gates (multi-metric + valcum)
  - Builds the selection pool (legacy-locked to “survivors”)
  - Calls the selector to pick champions

- `posthoc_gates.py`
  - Computes thresholds and applies boolean masks:
    - `passes_gates`
    - `valcum_pass`
    - (optional) `is_pareto`
  - Returns threshold tables for reporting and debugging

- `posthoc_selection.py`
  - Receives the (already gated) pool
  - Sorts deterministically
  - Picks champions per well/architecture/strategy with stable tie-breaks
  - Never uses `test_*` columns

Supporting utilities:
- `sort_utils.py` ensures stable ordering and deterministic tie-breaking.

---

### D) Validation layer (re-run champions, compare, audit)
**Purpose:** check whether the chosen champions remain good when retrained / re-executed.

- `validation_suite.py` defines batches of experiments
- `pivot_validation.py` prepares files/artifacts and runs validations
- `validation.py` compares HPO vs validation results and produces delta/equivalence outputs

Important rule:
- **TEST metrics are audit-only** (reported after selection, never used to select).

---

## Core contracts (what never changes)

### Anti-leak rule
- Any `test_*` column is **dropped before gating/selection**.
- TEST is used only to compute audit diagnostics after picks are made.

### Deterministic selection
- Sorting is stable.
- Ties are broken using consistent identifier columns when available.
- Same inputs → same champions.

### Single source of truth for the decision metric
- The “decision column” is resolved from config (`selection_col` or `metric_to_optimize`).
- Its direction (lower/higher is better) is resolved once via `resolve_ordering`.

---

## Two selection modes (conceptual)

### Legacy mode (gates + pick from survivors)
- Filters candidates with quantiles/MAD/valcum (VAL-only).
- Selection pool is **survivors** (passed gates).
- Picks top candidates per well using stable sorting.
- Simple, fast, very auditable.

### Neighborhood mode (robust local scoring)
- Builds a pool (e.g., top-pct or band on VAL).
- Computes local robustness scores based on neighborhood behavior (VAL-only).
- Picks winners based on robust local criteria.
- TEST remains audit-only after selection.

(If your current run is locked to legacy survivors-only, the behavior follows the first mode.)

---

## Typical user workflows

### Workflow 1 — “Run HPO and pick champions”
1. Run campaign(s) (Optuna or static profile) → generate leaderboards.
2. Aggregate leaderboards into a master dataframe.
3. Run post-hoc filtering:
   - gates → survivors → champions
4. Export champions for deployment.

### Workflow 2 — “Validate champions”
1. Generate a validation profile from selected champions/top trials.
2. Execute validation campaigns (retrain/re-run).
3. Compare HPO vs validation metrics and produce reports:
   - deltas, equivalence flags, audit stats.

### Workflow 3 — “Cross-family report”
1. Canonicalize results from Seq2/Arps/Darts.
2. Produce unified tables:
   - top-N per well
   - architecture summaries (mean/IQR)
   - consolidated leaderboards

---

## Outputs you can expect

- **Champion table**: one (or few) best trial(s) per well / architecture / strategy
- **Thresholds table**: quantile cutoffs and MAD bounds per well/metric
- **Diagnostics tables**: pool sizes, survivors counts, selection notes
- **Validation reports**: equivalence and delta tables between HPO and validation runs
- **Canonical reports**: cross-family comparison tables (optional)

---

## If you’re new to the codebase

Start from:
- `hpo_runner.py` (how experiments run and where results are written)
- `posthoc_filtering.py` (the practical entry point for champion selection)
- `posthoc_gates.py` + `posthoc_selection.py` (how filtering and selection are done safely)

That’s enough to understand the system end-to-end without reading every function.


### **Summary of Documentation Structure**

1.  **Core Definitions & Configuration (Iter 1):**
    *   `posthoc_config.py`: Configuration schemas and resolution logic.
    *   `score_ordering.py`: Metric directionality rules.
    *   `score_semantics.py`: Column naming and meaning definitions.

2.  **Mechanisms & Standards (Iter 2):**
    *   `posthoc_gates.py`: Statistical filtering logic (quantiles, MAD).
    *   `sort_utils.py`: Deterministic sorting helpers.
    *   `selection_contract.py`: Standardized output schemas.

3.  **Execution & Orchestration (Iter 3):**
    *   `posthoc_filtering.py`: Legacy path orchestrator.
    *   `neighborhood_selection.py`: Neighborhood path logic.
    *   `posthoc_selection.py`: Legacy selection strategies.

4.  **Validation Ecosystem (Iter 4):**
    *   `validation.py`: HPO vs. Validation comparison logic.
    *   `validation_suite.py`: Multi-experiment suite definitions.
    *   `pivot_validation.py`: Low-level validation execution engine.

5.  **Input & Execution Core (Iter 5):**
    *   `hpo_runner.py`: Main campaign execution engine.
    *   `search_space.py`: Hyperparameter space definitions.
    *   `optuna_utils.py`: Optuna integration helpers.
    *   `profile_generators.py`: Static grid generation.
    *   `final_suite.py`: Cross-family canonicalization and reporting.

This structure logically follows the flow of data through the system:
**Definition** $\rightarrow$ **Input/Execution** $\rightarrow$ **Filtering/Selection** $\rightarrow$ **Validation** $\rightarrow$ **Reporting**.


# HPO Core Definitions and Configuration

This documentation covers the foundational modules of the HPO analysis pipeline. These files define the configuration structures, rules for metric sorting, and semantic definitions for scoring columns. They serve as the "source of truth" for downstream filtering and selection processes.

## 1. `posthoc_config.py`

This module manages the configuration objects that control the post-hoc analysis and filtering pipeline. It ensures that user inputs are normalized and that critical parameters (like which metric to optimize) are resolved deterministically.

### Key Classes & Functions

*   **`PostHocConfig` (Data Class)**:
    *   Holds low-level filtering parameters: list of `metrics`, direction dictionaries (`lower_is_better`), quantile thresholds (`primary_quantile`), MAD guard settings, and Pareto front toggles.
    *   Defines column names for grouping (`well_col`, `arch_col`) and strategy identification.

*   **`SelectionRunConfig` (Data Class)**:
    *   Holds high-level execution settings: input/output directories (`results_dir`), the chosen `selector_mode` (e.g., `LEGACY_WEIGHTED` vs. `NEIGHBOR_TOP_PCT`), and visualization toggles.

*   **`make_default_config(**kwargs) -> PostHocConfig`**:
    *   **Sanitization:** Filters out unknown keyword arguments to prevent runtime errors.
    *   **Resolution:** Determines the **Single Source of Truth** for the decision column. It resolves conflicts between `selection_col` (explicit) and `metric_to_optimize` (legacy), ensuring both point to the same metric.
    *   **Consistency:** Automatically updates `lower_is_better` and `primary_quantile` defaults to ensure the chosen decision metric is actually being filtered correctly.

---

## 2. `score_ordering.py`

This module defines the directionality of optimization for metrics (i.e., whether a lower value or a higher value is "better"). It prevents accidental sorting inversions during champion selection.

### Key Classes & Functions

*   **`Ordering` (Data Class)**:
    *   A simple container storing the `metric` name, `ascending` bool (True if lower is better), `higher_is_better` bool, and the `reason` for this determination.

*   **`resolve_ordering(...) -> Ordering`**:
    *   The central logic for determining sort direction.
    *   **Priority 1:** Uses the `lower_is_better` dictionary provided in the config.
    *   **Priority 2:** Applies hard-coded invariants for known system metrics (e.g., `weighted_score` and `robust_score` are *always* minimized).
    *   **Fallback:** Defaults to ascending (lower is better) with a warning log.

*   **`analyze_across_campaigns(...)`**:
    *   A high-level utility that takes a master dataframe, resolves the ordering using the function above, and prints the top performing models across all campaigns.

---

## 3. `score_semantics.py`

This module handles the naming conventions and "meaning" behind score columns. It is particularly useful for distinguishing between different versions of "robust scores" (Legacy vs. Neighborhood) and ensuring backward compatibility via column aliasing.

### Key Classes & Functions

*   **`ScoreSemantics` (Data Class)**:
    *   Metadata object containing the `kind` of score (e.g., "legacy", "neighbor"), the `requested_col` name, and the `canonical_col` name.

*   **`resolve_score_semantics(score_col, ...) -> ScoreSemantics`**:
    *   Analyzes a requested column name. If `robust_score` is requested, it determines the underlying implementation/kind based on the pipeline context.

*   **`ensure_semantic_score_columns(df, semantics) -> DataFrame`**:
    *   Ensures that the DataFrame contains the necessary column aliases.
    *   *Example:* If the user requests `robust_score` (legacy), this function ensures a column named `legacy_robust_score` exists (copying data if necessary) so downstream selectors can find it reliably.

---

## Module Interaction & Communication

These three modules work together to establish the "ground rules" before any data processing happens:

1.  **Configuration (`posthoc_config.py`)**: The pipeline starts by creating a configuration object. It decides **WHAT** metric is the target (e.g., `"weighted_score"`).
2.  **Semantics (`score_semantics.py`)**: The pipeline checks the dataframe to ensure the target metric exists. If the target is a complex type (like a robust score), `score_semantics` ensures the column is correctly aliased so the code knows **WHICH** version of the score to use.
3.  **Ordering (`score_ordering.py`)**: Once the target metric is identified and located, `score_ordering` is called to determine **HOW** to sort it. It uses the settings from `posthoc_config` to enforce the correct sorting direction (ascending/descending).

**Flow:**
`Config` defines the target $\rightarrow$ `Semantics` validates/aliases the target column $\rightarrow$ `Ordering` defines the sort direction.

# HPO Mechanisms and Standards

This section details the operational logic of the HPO pipeline. These modules implement the mechanisms for filtering candidates, sorting them deterministically, and enforcing a standardized data contract for the final output.

## 1. `posthoc_gates.py`

This module provides the low-level logic for calculating statistical thresholds and applying filters (gates) to the trial data. It is stateless, operating purely on DataFrames based on the provided configuration.

### Key Functions

*   **`quantile_thresholds(df, cfg) -> DataFrame`**:
    *   Calculates simple quantile cutoffs (e.g., "top 20%") for each metric defined in the config, grouped by well.
    *   Returns a dataframe of threshold values.

*   **`mad_guards(df, cfg) -> DataFrame`**:
    *   Computes robust outlier bounds using the **Median Absolute Deviation (MAD)**.
    *   Supports one-sided or two-sided bounds and log-transformation for skewed distributions.
    *   Returns lower/upper bounds per metric/well to filter out "too good to be true" or anomalous trials.

*   **`apply_valcum_gate(df, cfg) -> DataFrame`**:
    *   Applies a specific band filter on the cumulative validation error (`val_smape_cum`).
    *   Uses adaptive logic (smooth, rule-based, or strict) to determine the lower bound (`q_low`), preventing the selection of trials that are statistically implausible.
    *   Adds a boolean column `valcum_pass`.

*   **`compose_predicates(...)` & `apply_multi_metric_gates(...)`**:
    *   Combines the calculated quantile and MAD thresholds into executable filter functions.
    *   Vectorizes the application of these filters across the dataframe to produce a single `passes_gates` boolean mask.

---

## 2. `sort_utils.py`

This module ensures that sorting operations are **stable** and **deterministic**. Standard pandas sorting can be non-deterministic when ties occur, which is problematic for reproducibility in HPO.

### Key Functions

*   **`infer_tiebreak_cols(df) -> List[str]`**:
    *   Scans the dataframe for unique identifiers (e.g., `trial_id`, `job_hash`, `optuna_trial_number`).
    *   Returns a list of columns to use as secondary sort keys to break ties.

*   **`stable_sort(df, metric, ordering, ...) -> DataFrame`**:
    *   Sorts the dataframe by the primary `metric`.
    *   Respects the direction defined in the `Ordering` object (ascending vs. descending).
    *   **Crucially**, appends the tie-break columns to the sort keys to guarantee a fixed order every time.

*   **`topk(...)`**:
    *   A convenience wrapper around `stable_sort` that returns the top $K$ rows.

*   **`best_idx_by_group(...)`**:
    *   Finds the index of the single best row per group (e.g., per well) using stable sorting logic, avoiding the potential randomness of `idxmin`/`idxmax`.

---

## 3. `selection_contract.py`

This module enforces the **Interface Contract** for the selection process. Regardless of the complex logic used upstream, the final output must adhere to a strict schema to ensure compatibility with downstream reporting tools.

### Key Concepts

*   **Canonical Schemas**: Defines fixed column lists for:
    *   `CANONICAL_SUMMARY_COLUMNS`: High-level metrics for the chosen champion.
    *   `CANONICAL_DIAGNOSTICS_COLUMNS`: Metadata about how the selection was made (pool size, mode, etc.).
    *   `CANONICAL_THRESHOLDS_COLUMNS`: The specific cutoff values used.

### Key Functions

*   **`make_selection_result_contract(...) -> Dict`**:
    *   The primary factory function. It takes raw results (top performers, diagnostics) and packages them into a standardized dictionary.
    *   Ensures all required keys (`top_performers`, `summary`, `thresholds`, `meta`) exist and contain DataFrames with the correct columns.

*   **`build_canonical_summary(...)`**:
    *   Constructs the summary table.
    *   Calculates **audit-only** metrics like `regret_test` (difference between chosen model and theoretical best in pool) and `val_test_spearman` (correlation).
    *   **Note:** This function handles test data for *reporting* only; it does not influence the selection itself.

*   **`build_legacy_regret_diagnostics(...)`**:
    *   A patch function to compute "regret" metrics for the legacy pipeline path, ensuring it provides comparable audit data to newer selection methods without breaking the "no test data in selection" rule.

---

## Module Interaction & Communication

These modules form the "engine room" of the pipeline:

1.  **Gating (`posthoc_gates.py`)**: Data enters and is annotated with pass/fail masks (`passes_gates`, `valcum_pass`) based on statistical thresholds.
2.  **Sorting (`sort_utils.py`)**: The filtered data is sorted deterministically using `stable_sort`, ensuring that repeated runs yield identical results even when scores are tied.
3.  **Contract (`selection_contract.py`)**: Finally, the selected champions and their metadata are passed to `make_selection_result_contract`. This function cleans, validates, and standardizes the output, guaranteeing that the final result is safe for consumption by the reporting layer.

**Flow:**
`Gates` filter data $\rightarrow$ `SortUtils` orders candidates $\rightarrow$ `Contract` packages the result.


# HPO Execution & Orchestration

This section documents the execution layer, where the abstract definitions and mechanisms are orchestrated to perform the actual champion selection. The pipeline supports two distinct paths: a **Legacy** path (based on strict gating) and a **Neighborhood** path (based on robust local scoring).

## 1. `posthoc_filtering.py` (Legacy Orchestrator)

This module acts as the main entry point for the **Legacy** filtering pipeline. It chains together data preparation, threshold calculation, gating, and final selection into a cohesive workflow.

### Key Functions

*   **`run_distribution_filter(master_df, cfg) -> Dict`**:
    *   **Orchestration:** Calls `sanitize_and_validate`, `deduplicate_by_signature`, `quantile_thresholds`, `mad_guards`, `apply_multi_metric_gates`, and `select_champions_by_strategy` in sequence.
    *   **Anti-Leak Enforcement:** Explicitly drops `test_*` columns before passing data to any filtering logic.
    *   **Scoring Injection:** If `robust_score` is requested, it ensures the score is calculated (`plug_robust_score`) and configures the pipeline to optimize for it.
    *   **Output:** Returns a dictionary containing the final `top_performers`, the `thresholds` used, and intermediate dataframes (`survivors`, `gate_counts`) for auditing.

*   **`deduplicate_by_signature(...)`**:
    *   Ensures that identical hyperparameter configurations (same "signature") are not treated as distinct candidates.
    *   Keeps only the best performing trial per signature to prevent a single good configuration from flooding the top-k list.

*   **`sanitize_and_validate(...)`**:
    *   Prepares the raw dataframe by ensuring required columns exist and converting metrics to numeric types, handling missing values gracefully.

---

## 2. `neighborhood_selection.py` (Neighborhood Logic)

This module implements the **Neighborhood Selection** strategy, an alternative to strict gating. Instead of binary pass/fail gates, it defines a local "neighborhood" of similar models and scores candidates based on their performance relative to that neighborhood.

### Key Concepts

*   **Neighborhoods:** Defined dynamically based on proximity in feature space (e.g., similar `learning_rate`, `epochs`, `batch_size`).
*   **Robust Scoring:** Calculates a score that penalizes:
    *   **Instability:** High local variance (`local_mad`).
    *   **Inconsistency:** Large gaps between validation metrics (`rank_gap`).
    *   **"Luck":** Performance that is statistically unlikely given the neighborhood distribution.

### Key Functions

*   **`run_neighborhood_selection(...) -> (DataFrame, DataFrame)`**:
    *   **Orchestrator:** Manages the selection process for this path.
    *   **Grouping:** Iterates through data grouped by dataset/well/architecture.
    *   **Selection:** Calls `_basic_select` for each group to find the best candidate.
    *   **Audit:** Calculates "regret" metrics (comparing the chosen model against the theoretical best in the pool) using test data *after* selection is made.

*   **`compute_robust_scores(...)`**:
    *   The core math engine. For a set of candidates, it identifies neighbors, calculates robust statistics (MAD, IQR), and computes the penalized score.

*   **`pick_pool_idx(...)`**:
    *   Selects the initial pool of candidates to evaluate (e.g., the top 20% by validation score) before applying the expensive robust scoring logic.

---

## 3. `posthoc_selection.py` (Legacy Selection Logic)

This module handles the final step of the **Legacy** pipeline: picking the "winner" from the pool of survivors that passed all gates.

### Key Functions

*   **`select_champions_by_strategy(df, cfg) -> DataFrame`**:
    *   **Strategy Dispatch:** Supports different selection strategies (e.g., "best of the best", "median of the best").
    *   **Grouping:** Applies selection logic per-group (well/architecture).
    *   **Relaxation:** If the strict gating produced an empty pool, it calls `_filter_pool_with_relaxation` to progressively loosen constraints until candidates are found.

*   **`_filter_pool_with_relaxation(...)`**:
    *   **L0 (Strict):** Requires passing all gates (quantiles + MAD + cumulative band) + Pareto optimality.
    *   **L1 (Relaxed):** Requires passing only the primary gates.
    *   **L2 (Loose):** Requires passing only the cumulative band gate.
    *   **L3 (Fallback):** Returns the raw pool if all filters fail.

*   **`_parse_config(...)`**:
    *   Translates the high-level `PostHocConfig` into a specialized `SelectionConfig`, resolving the final sort column and direction based on the "Single Source of Truth" rules defined in Iteration 1.

---

## Module Interaction & Communication

1.  **Dispatcher:** The main `analysis.py` (from Iter 4) will decide whether to call `run_distribution_filter` (Legacy) or `run_neighborhood_selection` (Neighborhood).
2.  **Legacy Path:**
    *   `run_distribution_filter` (in `posthoc_filtering.py`) uses `posthoc_gates.py` (Iter 2) to filter data.
    *   It then passes the "survivors" to `select_champions_by_strategy` (in `posthoc_selection.py`) to pick the final winners.
3.  **Neighborhood Path:**
    *   `run_neighborhood_selection` (in `neighborhood_selection.py`) is self-contained. It selects a pool, scores candidates relative to their neighbors, and picks a winner directly.
4.  **Commonality:** Both paths rely on the definitions from Iter 1 (`PostHocConfig`, `resolve_ordering`) to ensure they are optimizing for the correct metric in the correct direction.

**Flow (Legacy):**
`run_distribution_filter` $\rightarrow$ `gates` $\rightarrow$ `survivors` $\rightarrow$ `select_champions_by_strategy` $\rightarrow$ `champions`

**Flow (Neighborhood):**
`run_neighborhood_selection` $\rightarrow$ `pool` $\rightarrow$ `compute_robust_scores` $\rightarrow$ `champions`


# HPO Validation Ecosystem

This section documents the validation framework, which ensures that the "champion" models selected by the HPO pipeline perform reliably when retrained, potentially under different conditions (e.g., different seeds, aggregation methods, or ensemble sizes).

## 1. `validation.py`

This module provides the core logic for comparing the original HPO results against the new "Validation" run results. It focuses on generating equivalence reports to verify if the retrained models match their HPO performance.

### Key Functions

*   **`run_full_validation_workflow(...)`**:
    *   **Orchestrator:** Automates the entire loop: preparing paths/configs $\rightarrow$ generating a profile from the top N HPO trials $\rightarrow$ executing the validation campaign $\rightarrow$ generating the comparison report.
    *   **Replica Logic:** It explicitly replicates the pathing logic used in successful notebooks to ensure the validation run finds the correct data and profiles.

*   **`run_validation_comparison(...)`**:
    *   **Entry Point:** The main function called by users to compare two dataframes (HPO vs. Validation).
    *   **Auto-Merge:** If the input dataframes aren't already merged, it joins them on a "fingerprint" of hyperparameters (e.g., `learning_rate`, `batch_size`, `architecture`).
    *   **Output:** Returns a styled dataframe highlighting performance deltas.

*   **`build_validation_delta_table(...)`**:
    *   **Math Engine:** Calculates the absolute percentage-point difference (`delta_cum_pp`) between HPO and Validation metrics.
    *   **Equivalence Check:** Flags a model as "Equivalent" if the delta is within a strict tolerance band (e.g., $\pm 0.25$ pp).

---

## 2. `validation_suite.py`

This module enables running a **suite** of validation experiments. Instead of just one validation run, users can define multiple experiments (e.g., "Standard Reconstruction", "Warm Start", "Raw History") to test how robust the champions are to different deployment strategies.

### Key Classes

*   **`ValidationExperiment` (Data Class)**:
    *   Defines a single experiment configuration, including the `policy` (e.g., `'reconstruct'`, `'hp_raw'`) and any specific overrides.
    *   Generates unique run names and CSV filenames to keep results organized.

### Key Functions

*   **`run_validation_suite(...)`**:
    *   **Batch Processor:** Iterates through a list of `ValidationExperiment` objects.
    *   **Execution:** For each experiment, it creates the necessary configuration overrides and calls `run_pivot_validation`.
    *   **Persistence:** Saves a "minimal CSV" for each experiment containing just the essential metrics, making it easy to plot/compare results across the entire suite later.

---

## 3. `pivot_validation.py`

This is the low-level orchestrator that bridges the gap between the abstract "validation intent" and the actual execution engine (`run_campaign`). It handles the messy details of file I/O, configuration patching, and artifact management.

### Key Functions

*   **`run_pivot_validation(...)`**:
    *   **The Engine:** The workhorse function. It loads the champions profile, applies any requested filters (e.g., "only Seq2Seq models"), prepares the YAML configuration and CSV artifacts, and launches the training job (either interactively or via subprocess).

*   **`prepare_run_artifacts(...)`**:
    *   **Setup:** Creates the specific YAML config file and CSV profile required by the training runner.
    *   **Sanitization:** Calls `sanitize_profile_for_execution` to remove any result columns (like `val_smape`) from the input profile, ensuring the training run is not contaminated by previous results.

*   **`_compare_with_hpo(...)`**:
    *   **Smart Merge:** An intelligent helper that attempts to match validation results back to the original HPO master leaderboard.
    *   **Fingerprinting:** It dynamically selects the best set of columns ("fingerprint") to join on, handling minor floating-point differences and schema variations between architectures (e.g., `Seq2` vs `Darts`).

## Module Interaction & Communication

1.  **Definition:** The user defines a list of `ValidationExperiment`s in `validation_suite.py`.
2.  **Orchestration:** `validation_suite.py` loops through these experiments and calls `run_pivot_validation` (in `pivot_validation.py`) for each one.
3.  **Execution:** `pivot_validation.py` prepares the files on disk and calls the main training runner.
4.  **Analysis:** Once training is done, `validation.py` is used (often via `_compare_with_hpo`) to mathematically compare the new results against the original HPO baseline, checking for reproducibility.

**Flow:**
`Suite` defines experiments $\rightarrow$ `Pivot` executes runs $\rightarrow$ `Validation` compares results.

# HPO Input & Execution Core

This section documents the foundational components that define **what** to optimize and **how** to execute it. These modules handle hyperparameter search spaces, integration with Optuna, profile generation for experiments, and the main execution engine.

## 1. `hpo_runner.py`

This is the primary execution engine for HPO campaigns. It orchestrates the entire lifecycle of an experiment, from reading the configuration to running jobs and collating results.

### Key Functions

*   **`run_campaign_from_config_object(config, profile_override_path) -> DataFrame`**:
    *   **Orchestrator:** The main entry point. It determines if the run is a full HPO loop (with optimization cycles), a "straight run" (using a pre-defined profile like validation), or a "micro-campaign" (for grid searches).
    *   **Cleanup:** Automatically cleans up old study databases and artifacts to ensure a fresh run.
    *   **Execution:** Dispatches to `run_hpo_campaign_loop` (for optimization) or `run_robust_pipeline` (for static profiles).

*   **`run_robust_pipeline(config, profile_path) -> str`**:
    *   **Pipeline Execution:** Runs the core forecasting pipeline for a given set of trials (profile).
    *   **Job Generation:** Converts the profile CSV into executable job dictionaries.
    *   **Artifact Sync:** Ensures that results from distributed workers are correctly gathered and synced back to the intended output directory.

*   **`run_hpo_campaign_loop(...)`**:
    *   **The Loop:** Implements the iterative HPO process:
        1.  Ask Optuna for N new trials.
        2.  Run the pipeline for these trials.
        3.  Collate metrics (`val_smape`).
        4.  Report results back to Optuna to refine the next batch.
    *   **Multi-Objective Support:** Handles both single-objective (weighted score) and multi-objective optimization (Pareto front).

---

## 2. `search_space.py`

This module defines the hyperparameter search spaces for different model architectures. It acts as the "gene pool" for Optuna.

### Key Functions

*   **`define_seq2pin_family_space(...)`**:
    *   Defines search ranges for Seq2Seq / PINN models (e.g., `learning_rate` log-uniform, `hidden_size` categorical).
    *   Uses `_suggest_rounded_float` to ensure parameters are serializable and reproducible.

*   **`define_darts_space(...)`**:
    *   A unified search space for Darts models (TiDE, NHiTS, etc.).
    *   **Two-Stage Strategy:** First picks a model type (`physics_strategy`), then picks a specific profile configuration for that model, and finally samples universal training knobs (`epochs`, `batch_size`).

*   **`define_arps_space(...)`**:
    *   Defines the space for Arps decline curve models, handling both grid-based and solver-based parameterization.

*   **`SEARCH_SPACE_MAPPING`**:
    *   A dictionary mapping architecture names (e.g., "Seq2PIN", "Darts") to their respective search space functions, used by the runner to dispatch the correct logic.

---

## 3. `optuna_utils.py`

This module provides helper functions to interact with the Optuna library, handling study creation, trial generation, and result reporting.

### Key Functions

*   **`generate_trials_from_study(...)`**:
    *   **Trial Factory:** Creates an Optuna study (or loads an existing one) and asks the sampler for $N$ new configurations to try.
    *   **Persistence:** Saves these configurations to a CSV file ("profile") that the pipeline runner can execute.

*   **`report_results_to_study(...)`**:
    *   **Feedback Loop:** Reads the `leaderboard.csv` from a completed cycle and updates the Optuna study with the observed metrics.
    *   **Objective Handling:** Supports reporting a single scalar (e.g., `weighted_score`) or multiple values for multi-objective optimization.

*   **`generate_profile_from_top_trials(...)`**:
    *   **Harvesting:** Extracts the best performing trials from a study to create a "Validation Profile".
    *   **Pareto Handling:** For multi-objective studies, it intelligently selects a diverse set of trials from the Pareto front using a maximin distance strategy.

---

## 4. `profile_generators.py`

This module contains utilities for creating static experiment profiles, useful for grid searches or manual experiments where Optuna is not needed.

### Key Functions

*   **`generate_grid_search_profile(...)`**:
    *   **Cartesian Product:** Takes a dictionary of parameters and generates a full grid of all possible combinations.
    *   **Output:** Saves the resulting list of experiments to a YAML file, ready to be run by the pipeline.

---

## 5. `final_suite.py`

This module provides the infrastructure for "Final Suites" — standardized comparison runs across different model families (Seq2Seq, Arps, Darts). It handles data normalization (canonicalization) to ensure apples-to-apples comparison tables.

### Key Functions

*   **`canonicalize_seq2 / arps / darts`**:
    *   **Normalization:** Takes raw result CSVs from different families and maps their specific columns (e.g., `n_epochs` vs `epochs`, `variant` vs `physics_strategy`) to a standard `CANONICAL_SCHEMA`.

*   **`build_canonical_validation_df(...)`**:
    *   **Assembly:** Loads and merges results from multiple families into a single master dataframe for reporting.

*   **`build_topn_per_well_view(...)`**:
    *   **Ranking:** Generates a "Top-N" leaderboard per well, correctly ordering models (e.g., prioritizing PINN over others if configured) and presenting clean, readable metrics.

*   **`build_architecture_summary_iqr(...)`**:
    *   **Aggregated Report:** Creates a high-level summary table showing the Mean and IQR (Interquartile Range) of performance metrics for each architecture, perfect for final reports or papers.

## Module Interaction & Communication

1.  **Definition:** `search_space.py` defines *what* parameters can vary. `profile_generators.py` can create static sets of these.
2.  **Trial Generation:** `optuna_utils.py` uses the search spaces to generate dynamic batches of trials (CSV profiles).
3.  **Execution:** `hpo_runner.py` reads these profiles, configures the pipeline, and executes the jobs.
4.  **Feedback:** `optuna_utils.py` reports the results back to the study.
5.  **Final Reporting:** `final_suite.py` takes the final results from all campaigns, normalizes them, and produces the consolidated comparison tables.

**Flow:**
`Search Space` $\rightarrow$ `Optuna Utils` (Generate) $\rightarrow$ `HPO Runner` (Execute) $\rightarrow$ `Optuna Utils` (Report) $\rightarrow$ `Final Suite` (Compare)