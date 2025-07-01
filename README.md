# Equinor\_Forecast — An Adaptive Framework for Energy-Sector Time-Series Prediction

## Method Overview

**Equinor\_Forecast** provides a unified workflow for forecasting production in oil, gas, and renewable-energy systems.
At its core is a *few-shot, continuously learning* algorithm that:

* learns from as few as 150 historical samples,
* updates itself each time new data arrive, and
* generates **medium-term cumulative forecasts** (8–16 weeks) that remain stable even when operating conditions shift.

The framework wraps this method in an end-to-end, open-source pipeline that also benchmarks modern deep architectures (e.g., N-Beats, NHiTS, TiDE) and classical baselines (ARIMA, AutoARIMA). Supported datasets range from Equinor’s Volve field to synthetic UNISIM reservoir simulations and Open Power System Data (OPSD) for wind, solar, and load.

---

## 1 Motivation

Industrial energy assets rarely supply the large, clean histories demanded by many machine-learning models. Long gaps between data acquisition and deployment can postpone value generation for years \cite{data\_random\_2017}.
To address this reality we pursue two design principles:

1. **Online adaptation** – retrain the model incrementally so that yesterday’s production instantly informs today’s forecast.
2. **Cumulative targets** – predict aggregated sums rather than noisy instantaneous rates, providing planners with smoother, more actionable signals \cite{lim\_time\_series\_2021}.

---

## 2 Data Transformation & Normalization

### 2.1 Cumulative Target Construction

Daily rates $y_t$ are first integrated to obtain the cumulative series.
Figure \ref{fig\:preprocessing} contrasts a raw oil-rate trace with its cumulative counterpart, revealing the latter’s reduced variance.

### 2.2 Lagged Feature Matrix

A sliding window of the previous seven days supplies the input vector

$$
\mathbf{X}_t=\bigl[X_{t-7,k},X_{t-6,k},\dots,X_{t-1,k}\bigr],\qquad k\in\mathcal S,
\tag{1}\label{eq:extended_feature_vector}
$$

and the stacked window

$$
\mathcal X_t=\bigl[\mathbf{X}_t,\mathbf{X}_{t-1},\dots,\mathbf{X}_{t-n}\bigr]
\tag{2}\label{eq:feature_matrix}
$$

feeds the learner.

### 2.3 Ratio Normalization

To prevent the cumulative target from drifting beyond the training convex hull, each label is scaled by a local mean:

$$
\tilde y_t=\frac{y_t}{\tfrac1n\sum_{i=1}^{n}X_{t-i,k}} .
\tag{3}\label{eq:normalized_response}
$$

The model operates on this dimension-less ratio, then re-multiplies by the same mean to recover physical units (Figure \ref{fig\:Norm}).

---

## 3 Online Training Loop

A **rolling window** of $W=150$ samples provides the daily training set

$$
\mathcal T_t=\bigl\{(x_{t'},y_{t'})\mid t-W\le t' < t\bigr\}.
\tag{4}\label{eq:sliding_window}
$$

After each new observation the model is fine-tuned and immediately used to forecast horizons
$h\in\{14,28,56,70,94,112\}$ days:

$$
\hat y_{t+h}=f_t(x_t).
$$

Fine-tune granularity is model-specific: deep learners update with the two most recent points; XGBoost refreshes on the latest 50.

Figure \ref{fig\:Slid} depicts the chronology, ensuring a gap of $h$ samples to avoid leakage.

---

## 4 Model Portfolio

| Category                      | Representative Models                       |
| ----------------------------- | ------------------------------------------- |
| **Few-shot online (default)** | XGBoost; custom DL architecture (Section 5) |
| Deep-learning benchmarks      | N-Beats, NHiTS, TiDE, TiDE + RIN, NLinear   |
| Statistical baselines         | ARIMA, AutoARIMA, Linear Regression         |
| Ensembles                     | Arbitrary combinations supported            |

All models share a standard interface (input 7, output 56, early stopping, adaptive LR) to enable fair evaluation.

---

## 5 Custom Deep Architecture & Ablation

The canonical network (Model 6) integrates:

1. **Transformer encoder** for long-range self-attention,
2. **Two 1-D convolutions** to capture local motifs,
3. **Bi-directional LSTM** for sequential context, and
4. **Dense head** with LeakyReLU.

Early stopping, dropout 0.3, and the Adam optimizer minimize

$$
\mathcal L=\frac1T\sum_{t=1}^{T}\bigl|y_t-f(\mathbf X_t;\boldsymbol\theta)\bigr|
\tag{5}\label{eq:objective_function_refined}
$$

(MAE).
Ablation progressively removes components (Models 1–5) and varies five hyperparameter “profiles” from *small-fast* to *large-robust*. Third-party baselines such as **The Golem** \cite{martinez\_golem:\_2023} and a stacked **GRU** \cite{werneck\_data\_driven\_2022} join the grid for completeness.
Figure \ref{fig\:Flow\_Volve} sketches Model 6.

---

## 6 Interpretability

A **longitudinal SHAP pipeline** accompanies training:

* every checkpoint exports mean absolute SHAP values,
* Gini and Spearman statistics monitor focus and stability,
* beeswarm plots reveal directionality.

Thus users obtain real-time insight into how feature influence evolves rather than a single post-hoc snapshot (Figure \ref{fig\:Pipeline}).

---

## 7 Evaluation & Metrics

Performance is reported for each horizon using

* **SMAPE** – scale-free primary score,
* **MAE** and **MSE** – absolute and squared errors,
* **Cumulative-error curves** – degradation with lead time.

Results are averaged over horizons (7 d → 112 d) to visualise the accuracy/lead-time trade-off.

---

## 8 Experimental Setup

### 8.1 Datasets

| Domain              | Dataset             | Key Details                              |
| ------------------- | ------------------- | ---------------------------------------- |
| Oil & gas           | **Volve** (Equinor) | 4 wells, daily, targets: BORE\_OIL\_VOL  |
| Synthetic reservoir | **UNISIM-II-H**     | 10 producers, 3 263 d, target: QOOB      |
| Renewables          | **OPSD** (GB)       | Wind (30 min → 1 d), Solar & Load (12 h) |

Minimal preprocessing: mean imputation, moving-window standardisation, optional retention of zeros (important for solar).

### 8.2 Hardware & Software

* AMD Ryzen 9 5900XT (16 cores, 32 threads)
* 31 GiB RAM, Manjaro Linux 6.6.65-1
* Parallel execution (up to 12 runs) with `forkserver`, in-memory caching, TensorFlow-XLA.

A full training–evaluation cycle takes ≈ 5 min; 12 experiments run concurrently.

---

## 9 Incremental vs Direct Cumulative Prediction

Two paradigms are compared (Table \ref{tab\:incremental\_vs\_direct}):

* **Incremental** – model predicts $y_{t+1}$; sums accumulate.
  *Pros*: stepwise interpretability.
  *Cons*: error drift.

* **Direct** – model outputs $Y_t$ outright.
  *Pros*: no compounding error.
  *Cons*: must extrapolate large values.

The online method adopts the **Direct** strategy, aided by the normalization in Eq. \eqref{eq\:normalized\_response} to keep extrapolation bounded.

---

## 10 Repository Map

```text
Equinor_Forecast/
├── notebooks/
│   ├── darts/            # Darts-based experiments
│   └── forecast/         # Custom pipelines
├── src/
│   ├── forecast_pipeline # Orchestrator & configs
│   ├── data/             # Loaders & preprocessing
│   ├── models/           # ML model implementations
│   ├── evaluation/       # Metrics & analysis
│   └── statistical/      # ARIMA, AutoARIMA, etc.
├── data/                 # Raw & processed datasets
├── experiments/          # Saved configs & results
└── output_manifest/      # Generated artefacts
```


## Overview

Equinor_Forecast is an end-to-end machine learning pipeline designed specifically for forecasting in the energy sector, with particular focus on oil well production prediction and renewable energy generation forecasting. The project implements and compares multiple state-of-the-art forecasting algorithms including modern deep learning architectures (N-Beats, NHiTS, TiDE, NLinear) and traditional statistical methods (ARIMA, AutoARIMA) to provide robust and accurate predictions for energy production optimization.

The framework supports multiple real-world energy datasets including the Volve oil field dataset from Equinor, UNISIM reservoir simulation data, and Open Power System Data (OPSD) for renewable energy forecasting. This comprehensive approach enables researchers and practitioners to evaluate and deploy forecasting models across different energy domains, from traditional oil and gas production to modern renewable energy systems.

## Repository Structure

The repository is organized into several key directories, each serving a specific purpose in the forecasting pipeline:

### Core Directories

- **`notebooks/`** - The execution hub containing all Jupyter notebooks for experiments and analysis
  - **`darts/`** - Notebooks implementing Darts library-based forecasting models
  - **`forecast/`** - Custom forecasting implementations and experimental notebooks
- **`src/`** - Source code containing the core forecasting pipeline and utilities
  - **`forecast_pipeline/`** - Main pipeline implementation with configuration and execution logic
  - **`common/`** - Shared utilities and common functionality
  - **`data/`** - Data loading and preprocessing modules
  - **`evaluation/`** - Model evaluation and metrics calculation
  - **`models/`** - Machine learning model implementations
  - **`prediction/`** - Prediction and inference modules
  - **`statistical/`** - Statistical forecasting methods
- **`data/`** - Dataset storage and data files
- **`experiments/`** - Experimental configurations and results
- **`output_manifest/`** - Output files and result manifests

### Model-Specific Directories

- **`VOLVE_MODELS/`** - Pre-trained models and configurations for Volve dataset
- **`UNISIM_MODELS/`** - Models trained on UNISIM reservoir simulation data  
- **`OPSD_MODELS/`** - Models for Open Power System Data renewable energy forecasting

## Main Notebooks

The notebooks directory contains the primary execution environment for the forecasting framework:

### Darts Implementation (`notebooks/darts/`)

- **`DARTS.ipynb`** - Comprehensive implementation of the Darts forecasting pipeline featuring multiple state-of-the-art models including TiDE, NLinear, N-Beats, NHiTS, and TiDE+RIN. This notebook provides an end-to-end workflow from data loading and preprocessing through model training, forecasting, and evaluation with support for multiple datasets and automated hyperparameter configuration.

- **`DARTS_Hybrid.ipynb`** - Advanced hybrid modeling approach combining multiple Darts models for improved forecasting accuracy through ensemble methods and model fusion techniques.

### Custom Forecasting Implementation (`notebooks/forecast/`)

- **`base_pipeline.ipynb`** - Foundation pipeline implementation providing the core framework for custom forecasting models and experimental setups.

- **`energy_based_forecast.ipynb`** - Specialized forecasting models designed specifically for energy sector applications, incorporating domain-specific features and physics-informed modeling approaches.

- **`forecast_DL.ipynb`** - Deep learning forecasting implementations featuring custom neural network architectures optimized for time-series prediction in energy applications.

- **`forecast_XGB.ipynb`** - XGBoost-based forecasting models providing gradient boosting solutions for time-series prediction with feature engineering and hyperparameter optimization.

- **`launch_jobs_wells.ipynb`** - Automated job execution system for running forecasting experiments across multiple oil wells and production scenarios.

- **`physics_feature_analysis.ipynb`** - Analysis of physics-based features and their impact on forecasting accuracy, incorporating domain knowledge from reservoir engineering and production optimization.

- **`results_analysis.ipynb`** - Comprehensive analysis and visualization of forecasting results, including performance comparisons, error analysis, and model interpretation.

- **`run_experiments.ipynb`** - Experimental execution framework for running systematic forecasting experiments across different models, datasets, and configurations.

- **`shap.ipynb`** - Model interpretability analysis using SHAP (SHapley Additive exPlanations) values to understand feature importance and model decision-making processes.

## Datasets

The framework supports three primary datasets representing different aspects of energy forecasting:

### Volve Oil Field Dataset

The Volve dataset represents real-world oil production data from Equinor's Volve oil field in the North Sea. This dataset includes production data from multiple wells with the following characteristics:

- **Wells**: 15/9-F14, 15/9-F12, 15/9-F11, and additional production wells
- **Target Variable**: BORE_OIL_VOL (oil production volume)
- **Data Location**: `data/volve/Volve_Equinor`
- **Features**: Production rates, pressure measurements, and operational parameters

### UNISIM Reservoir Simulation Data

UNISIM provides synthetic but realistic reservoir simulation data for testing and validation of forecasting models:

- **Production Wells**: Prod-1 through Prod-10, P16 (UNISIM-IV)
- **Target Variables**: Q00B (production rate), BORE_OIL_VOL
- **Data Locations**: 
  - `data/unisim/production.c`
  - `data/UNISIM-IV-2026/Well_IV.csv`
- **Features**: Simulated production data with controlled reservoir parameters

### Open Power System Data (OPSD)

OPSD provides renewable energy generation data for wind, solar, and load forecasting:

- **Energy Types**: Wind generation, Solar generation, Load demand
- **Target Variable**: GB_GBN_<type>_generation_actual
- **Data Location**: `data/OPSD/time_series_30`
- **Geographic Scope**: European power system data with high temporal resolution

## Algorithms & Methods

The framework implements a comprehensive suite of forecasting algorithms spanning traditional statistical methods to cutting-edge deep learning architectures:

### Deep Learning Models

**N-Beats (Neural Basis Expansion Analysis for Time Series)** - A pure deep learning approach for time-series forecasting that uses backward and forward residual links and a deep stack of fully-connected layers to achieve state-of-the-art performance without requiring feature engineering.

**NHiTS (Neural Hierarchical Interpolation for Time Series)** - An advanced neural network architecture that leverages hierarchical interpolation and multi-rate signal processing to capture both short-term and long-term temporal dependencies in time-series data.

**TiDE (Time-series Dense Encoder)** - A modern encoder-decoder architecture specifically designed for long-horizon forecasting that combines the benefits of linear models with the expressiveness of deep neural networks.

**TiDE+RIN (TiDE with Reversible Instance Normalization)** - An enhanced version of TiDE incorporating reversible instance normalization to improve model stability and forecasting accuracy across different scales and distributions.

**NLinear** - A simple yet effective linear model that serves as a strong baseline and often outperforms complex deep learning models on many time-series forecasting tasks.

### Traditional Statistical Methods

**ARIMA (AutoRegressive Integrated Moving Average)** - Classical statistical model for time-series forecasting that captures temporal dependencies through autoregressive and moving average components with differencing for stationarity.

**AutoARIMA** - Automated ARIMA model selection that uses statistical tests and information criteria to automatically determine optimal model parameters and seasonal components.

**Linear Regression** - Simple linear models adapted for time-series forecasting with engineered lag features and trend components.

### Ensemble and Hybrid Methods

The framework supports ensemble methods that combine predictions from multiple models to improve overall forecasting accuracy and robustness. The hybrid approaches leverage the strengths of both statistical and machine learning methods.

### Model Configuration and Training

The forecasting pipeline uses standardized configuration parameters across all models:

- **Input Chunk Length**: 7 time steps (lag window)
- **Output Chunk Length**: 56 time steps (forecast horizon)  
- **Training Size**: 150 + forecast horizon time steps
- **Early Stopping**: Implemented with patience for optimal model selection
- **Learning Rate Scheduling**: Adaptive learning rate adjustment during training
- **Cross-Validation**: Time-series aware validation splitting

### Evaluation Methodology

Model performance is evaluated using multiple metrics with particular emphasis on:

- **SMAPE (Symmetric Mean Absolute Percentage Error)** - Primary evaluation metric providing scale-independent performance assessment
- **MAE (Mean Absolute Error)** - Absolute error measurement for direct comparison
- **MSE (Mean Squared Error)** - Squared error metric emphasizing larger deviations
- **Cumulative Performance Analysis** - Long-term forecasting accuracy assessment

The evaluation framework includes comprehensive performance analysis across different datasets, wells, and forecasting horizons, with results visualized through detailed plots and statistical summaries. Model interpretability is enhanced through SHAP analysis, providing insights into feature importance and decision-making processes for improved understanding and trust in forecasting results.


