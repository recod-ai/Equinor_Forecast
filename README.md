# Equinor\_Forecast — An Adaptive Framework for Energy-Sector Time-Series Prediction

## Method Overview

**Equinor\_Forecast** is an open-source pipeline that delivers **medium-term cumulative forecasts** (8 – 16 weeks) for oil, gas and renewables.
Its few-shot algorithm

* trains on as little as **150 historical samples**,
* **updates continuously** as fresh data arrive, and
* maintains accuracy when operating conditions change.

The same framework benchmarks modern neural forecasters (N-Beats, NHiTS, TiDE) and statistical classics (ARIMA, AutoARIMA) on three public energy datasets: Volve (oil), UNISIM (synthetic reservoir) and OPSD (wind/solar/load).

---

## 1 Motivation

Most industrial assets lack the long, tidy histories that conventional ML expects. Waiting years to collect data delays value creation.
We therefore design for:

1. **Online adaptation** – daily fine-tuning turns yesterday’s measurement into today’s prior.
2. **Cumulative targets** – forecasting aggregated sums smooths noise and aids planning.

---

## 2 Data Transformation & Normalization

### 2.1 Cumulative Targets

Convert daily rates `y_t` into a running sum to reduce variance (see *Pre-processing* figure).

### 2.2 Lagged Feature Matrix

Create a seven-day sliding window:

```text
X_t = [ X_{t-7,k}, X_{t-6,k}, … , X_{t-1,k} ]   for k in S
X_mat_t = [ X_t , X_{t-1} , … , X_{t-n} ]
```

### 2.3 Ratio Normalization

To avoid out-of-range extrapolation, scale each cumulative label by a local mean:

```text
y_t_tilde = y_t / mean( X_{t-i,k} , i = 1 … n )
```

The model predicts the dimension-less `y_t_tilde`, then rescales it back to physical units.

---

## 3 Online Training Loop

With a rolling window of `W = 150` samples, the day-`t` training set is

```text
T_t = { (x_{t'}, y_{t'})  |  t-W ≤ t' < t }
```

After ingesting a new point, the model is fine-tuned and produces forecasts for horizons
`h ∈ {14, 28, 56, 70, 94, 112}` days:

```text
ŷ_{t+h} = f_t( x_t )
```

Deep models update on the two newest samples; XGBoost refreshes on the last 50.

---

## 4 Model Portfolio

| Category                      | Representative Models                     |
| ----------------------------- | ----------------------------------------- |
| **Few-shot online (default)** | XGBoost, custom DL architecture (Sec. 5)  |
| Deep-learning benchmarks      | N-Beats, NHiTS, TiDE, TiDE + RIN, NLinear |
| Statistical baselines         | ARIMA, AutoARIMA, Linear Regression       |
| Ensembles                     | Any combination                           |

All share the same data interface (7-step input, 56-step output) plus early stopping and adaptive learning rates.

---

## 5 Custom Deep Architecture & Ablation

**Model 6** (the default) stacks:

1. Transformer encoder for long-range context,
2. Two 1-D convolutions for local patterns,
3. Bi-directional LSTM for sequence memory,
4. Dense head with LeakyReLU.

Training minimises mean-absolute-error

```text
L = (1 / T) Σ | y_t − f( X_t ; θ ) |
```

Ablation tests five lighter variants (Models 1-5) and five hyper-parameter “profiles”. External baselines **The Golem** and a stacked **GRU** are also included.

---

## 6 Interpretability

A **longitudinal SHAP** routine runs in parallel:

* exports mean absolute SHAP values at every checkpoint,
* tracks focus (Gini) and stability (Spearman),
* plots beeswarm summaries.

Users therefore see how feature importance evolves over time, not just after training.

---

## 7 Evaluation & Metrics

For each horizon we report

* **SMAPE** (primary, scale-free),
* **MAE** and **MSE**,
* cumulative-error curves (accuracy vs lead-time).

Averaging across all horizons (7 – 112 days) highlights the accuracy/latency trade-off.

---

## 8 Experimental Setup

### 8.1 Datasets

| Domain              | Dataset         | Key Facts                               |
| ------------------- | --------------- | --------------------------------------- |
| Oil & gas           | **Volve**       | 4 wells, daily, target = BORE\_OIL\_VOL |
| Synthetic reservoir | **UNISIM-II-H** | 10 producers, 3 263 days, target = QOOB |
| Renewables          | **OPSD** (GB)   | Wind (30 min→1 d), Solar & Load (12 h)  |

Pre-processing: mean imputation, moving-window standardisation, optional zero retention (useful for solar).

### 8.2 Hardware & Software

* AMD Ryzen 9 5900XT (16 cores / 32 threads)
* 31 GiB RAM, Manjaro Linux 6.6.65-1
* Parallel pool (`forkserver`) + TensorFlow-XLA; 12 experiments run concurrently (≈ 5 min each).

---

## 9 Incremental vs Direct Cumulative Prediction

| Approach    | How It Works               | Pros                      | Cons                          |
| ----------- | -------------------------- | ------------------------- | ----------------------------- |
| Incremental | Predict `y_{t+1}` then sum | Stepwise interpretability | Error drift over time         |
| Direct      | Predict `Y_t` in one shot  | No compounding error      | Must extrapolate large values |

Equinor\_Forecast follows the **Direct** route, with ratio normalisation keeping predictions in-range.

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
  - **`statistical/`** - Statistical methods
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

Here’s a clear and concise summary in English of the ideas from your text:

---

**DATA PIPELINE**

The Data Pipeline is the foundation of the Equinor\_Forecast system, responsible for ingesting, processing, and preparing time series data from various energy sector sources. Its architecture is robust and scalable, ensuring data quality and consistency across heterogeneous datasets, including oil production, reservoir simulation, and renewable energy.

**Core Modules:**

* `data_loading.py`: Standardizes data ingestion from multiple sources, providing unified APIs for consistent data structures.
* `data_preparation.py`: Handles complex preprocessing for time series forecasting, including sliding window creation, feature engineering, and cumulative forecasting transformations.

**Supported Data Sources:**

* **Volve Oil Field Dataset:** Real-world oil production data, reflecting authentic industry patterns and noise.
* **UNISIM Reservoir Simulation:** Synthetic, controlled simulation data for validation and testing.
* **OPSD Renewable Energy Dataset:** Actual renewable generation data, introducing challenges like volatility and seasonality.

**Technical Highlights:**

* **Sliding Window Creation:** Uses configurable window sizes (default 7 days) for input features and flexible forecast horizons to prepare data for multi-step forecasting.
* **Feature Engineering:**

  * Temporal shift aligns targets with inputs.
  * Lagged and static feature construction captures dependencies.
  * Cumulative transformation smooths noise and aligns with total production forecasting needs.
  * Adaptive normalization improves generalization and prevents data leakage.

**Configuration and Extensibility:**

* Highly customizable via configuration for window sizes, forecast horizons, and normalization methods.
* Modular design allows easy integration of new data sources and custom preprocessing functions.
* Supports parallel processing for large-scale data.
* Centralized configuration, experiment tracking, and preprocessing version control ensure consistency and reproducibility.

**Usage:**
The pipeline is imported and used through simple Python functions, allowing straightforward integration into forecasting workflows.

![Pipeline Architecture](Data_Pipeline.png)

# Physics-Informed Neural Networks

## Overview

The PINN (Seq2Context model) represents the flagship architecture of the Equinor_Forecast system, implementing a sophisticated hybrid neural network that combines modern deep learning techniques with physics-informed strategies specifically designed for energy sector forecasting. This model addresses the unique challenges of cumulative production forecasting by integrating temporal sequence modeling with domain-specific knowledge about energy production physics.

The architecture represents a significant advancement in time series forecasting for energy applications, moving beyond traditional statistical methods and generic deep learning approaches to incorporate the fundamental physics governing energy production processes. The model's name "Seq2Context" reflects its core functionality: transforming input sequences into rich contextual representations that capture both temporal dependencies and physical constraints inherent in energy production systems.

The model's design philosophy centers on the recognition that energy production forecasting requires more than pattern recognition in historical data. Successful forecasting must account for the underlying physical processes, equipment constraints, reservoir characteristics, and operational parameters that drive production behavior. The Seq2Context architecture achieves this integration through a multi-branch design that processes temporal sequences through both data-driven neural networks and physics-informed computational pathways.

## Architectural Design

### Core Neural Network Components

The Seq2Context model implements a sophisticated multi-layer architecture that processes temporal sequences through several specialized components, each designed to capture different aspects of the forecasting problem.

**Transformer Encoder Block**: The foundation of the model's temporal processing capability lies in a transformer encoder block that implements multi-head self-attention mechanisms. This component excels at capturing long-range temporal dependencies that are crucial for understanding production trends over extended periods. The transformer architecture allows the model to selectively attend to relevant historical time points, automatically learning which past observations are most informative for predicting future production values.

The transformer encoder utilizes eight attention heads with a model dimension of 512, providing sufficient capacity to capture complex temporal relationships while maintaining computational efficiency. The multi-head attention mechanism enables the model to focus on different types of temporal patterns simultaneously, such as short-term fluctuations, medium-term trends, and long-term seasonal cycles that characterize energy production data.

Positional encoding is incorporated to provide the transformer with explicit temporal information, ensuring that the model understands the sequential nature of the input data. This is particularly important for energy forecasting applications where the timing of observations carries significant meaning for production planning and operational decision-making.

**Convolutional Layers**: Following the transformer encoder, the model employs two one-dimensional convolutional layers designed to extract local temporal patterns and features that complement the global attention mechanisms of the transformer. The first convolutional layer uses 64 filters with a kernel size of 3, capturing short-term temporal patterns and local variations in production rates.

The second convolutional layer expands to 128 filters while maintaining the same kernel size, allowing the model to learn more complex local patterns and feature combinations. These convolutional layers are particularly effective at identifying recurring patterns such as daily operational cycles, weekly maintenance schedules, or equipment-specific production signatures that appear consistently in energy production data.

Max pooling operations follow the convolutional layers to reduce temporal resolution while preserving the most important features. This dimensionality reduction improves computational efficiency and helps prevent overfitting by focusing on the most salient temporal patterns.

**Bidirectional LSTM Layer**: The model incorporates a bidirectional Long Short-Term Memory (LSTM) layer with 256 units to provide sequence memory and temporal context understanding. The bidirectional design processes sequences in both forward and backward directions, enabling the model to consider future context when interpreting past observations.

This bidirectional processing is particularly valuable for energy forecasting applications where production patterns often exhibit anticipatory behavior. For example, production rates might begin declining before scheduled maintenance periods, or increase in anticipation of favorable market conditions. The bidirectional LSTM captures these complex temporal relationships that unidirectional models might miss.

The LSTM layer includes dropout regularization with a rate of 0.3 to prevent overfitting and improve generalization to new production scenarios. The return_sequences parameter is set to True, ensuring that the full temporal sequence is preserved for downstream processing by the physics-informed components.

### Physics-Informed Strategy Integration

A distinguishing feature of the Seq2Context model is its integration of physics-informed strategies that incorporate domain knowledge about energy production processes. This integration is achieved through a specialized branch that operates in parallel with the neural network components, providing physics-based insights that complement data-driven learning.

**Physics Strategy Factory**: The model implements a factory pattern for creating and managing different physics-informed strategies. This design allows for flexible selection and combination of physical models based on the specific characteristics of the forecasting problem and the available domain knowledge.

The factory pattern provides several advantages for energy forecasting applications. It enables easy experimentation with different physical models, supports dynamic strategy selection based on data characteristics, and facilitates the integration of new physics-based approaches as they are developed. The factory also manages the parameterization of physical models, ensuring that reservoir properties, equipment specifications, and operational constraints are properly incorporated into the forecasting process.

**Exponential Decay Strategy**: This strategy implements exponential decline models commonly used in oil and gas production forecasting. The exponential decay model assumes that production rates decline exponentially over time according to the equation y(t) = y₀ * exp(-λt), where y₀ represents the initial production rate and λ represents the decline constant.

The exponential decay strategy is particularly effective for modeling the natural decline of oil wells due to reservoir pressure depletion. The model learns to estimate the decline parameters from historical data while incorporating physical constraints that ensure realistic decline behavior. This physics-informed approach provides more stable and interpretable forecasts compared to purely data-driven methods.

**Arps Decline Strategy**: The model implements the industry-standard Arps decline curves, which provide a more flexible framework for modeling production decline behavior. The Arps model supports three types of decline: exponential (b=0), hyperbolic (0<b<1), and harmonic (b=1), where b represents the decline exponent.

The Arps decline strategy automatically selects the most appropriate decline type based on the characteristics of the historical production data. This adaptive approach ensures that the physical model accurately represents the underlying reservoir and well behavior, leading to more accurate forecasts. The strategy also incorporates uncertainty quantification, providing confidence intervals around the decline curve predictions.

**Weighted Ensemble Strategy**: This strategy combines multiple physics-informed approaches using learned weights that reflect the relative importance of different physical models for the specific forecasting scenario. The ensemble approach recognizes that energy production systems often exhibit complex behavior that cannot be captured by a single physical model.

The weighted ensemble strategy learns optimal combination weights through the training process, automatically balancing the contributions of different physical models based on their predictive accuracy for the specific dataset. This adaptive weighting ensures that the most relevant physical insights are emphasized while less applicable models receive reduced influence.

**Static Pressure Strategy**: This strategy incorporates reservoir pressure information into the forecasting process, recognizing that production rates are fundamentally driven by pressure differentials between the reservoir and the wellbore. The static pressure strategy models the relationship between reservoir pressure and production rates, accounting for pressure depletion over time.

The strategy is particularly valuable for reservoir-level forecasting where pressure data is available from multiple wells or pressure monitoring systems. By incorporating pressure information, the model can better predict production behavior under different operational scenarios and reservoir conditions.

**Combined Exponential-Arps Strategy**: This hybrid strategy combines the simplicity of exponential decline models with the flexibility of Arps decline curves, providing a robust approach for handling diverse production scenarios. The combined strategy automatically selects the most appropriate model components based on the characteristics of the historical data and the forecasting requirements.

![PINNs Overview](PINN.png)
