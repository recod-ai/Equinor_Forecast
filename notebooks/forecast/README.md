# 📈 Adaptive Forecasting Pipeline for Online Training

This document outlines an advanced forecasting pipeline designed for **adaptive learning**. The system continuously updates its models as new data becomes available, making it ideal for time series where patterns evolve over time. The architecture is model-agnostic, supporting both **Deep Learning** (Keras/TensorFlow) and **Gradient Boosting** (XGBoost) backends within the same incremental fine-tuning framework.

---

### Core Concept: The Sliding Window Fine-Tuning Loop

The entire pipeline is built around a powerful online training loop. Instead of training a single static model, the system retrains or "fine-tunes" the model iteratively on new chunks of data.

**Visualized Process:**

1.  **Initial Training (Cold Start)**
    `[ Initial Historical Data ]` ➡️ 🚂 **Train Base Model**

2.  **Sliding Window Iteration**
    `[ Base Model ]` + `[ New Data Chunk 1 ]` ➡️ ✨ **Fine-tune & Predict**

3.  **Continuous Adaptation**
    `[ Fine-tuned Model ]` + `[ New Data Chunk 2 ]` ➡️ ✨ **Fine-tune & Predict**

This process repeats until all available data has been processed, ensuring the model's predictions are always based on the most recent information available.

---

### Pipeline Workflow at a Glance

The workflow is broken down into distinct, automated stages, from data ingestion to final evaluation.

| Step | Action | Description | Icon |
| :--- | :--- | :--- | :---: |
| **1. Data Loading** | `_load_data()` | Ingests and prepares time series from sources like Volve, UNISIM, or OPSD. Organizes wells for efficient processing. | 🗂️ |
| **2. Base Training** | `_train_base_models()` | Trains the initial, foundational model on a large, initial block of historical data. This model is saved and serves as the starting point for fine-tuning. | 🚂 |
| **3. Iterative Loop** | `_iterate()` | Manages the core sliding window. For each new time step, it prepares data chunks for the active wells. | 🔄 |
| **4. Fine-Tune & Predict** | `fine_tune_and_predict_well()` | **The core of the online learning.** This function is executed in parallel for multiple wells. It takes an existing model, fine-tunes it on the new data chunk, and generates a forecast. | ✨ |
| **5. Post-Processing** | `apply_filter_to_predictions()` | Applies an optional post-processing filter (e.g., Kalman Filter) to the raw predictions to smooth results and reduce noise. | 🔬 |
| **6. Final Evaluation** | `_final_eval()` | After all iterations are complete, it aggregates the step-by-step predictions and evaluates the model's overall performance against the true values. | 🏆 |

---

### Key Architectural Features

This pipeline is engineered for efficiency, scalability, and adaptability.

| Feature | Description |
| :--- | :--- |
| ✅ **Adaptive Learning** | Models are not static; they continuously evolve with new data, capturing drift and changing dynamics in the time series. |
| ✅ **Model Agnostic** | The same online training framework seamlessly supports different model families (`model_type="DL"` or `"XGB"`), allowing for flexible experimentation. |
| ✅ **Parallel Processing** | Leverages multiprocessing (`mp.Pool`) to fine-tune and predict for multiple wells simultaneously, dramatically reducing total execution time. |
| ✅ **Resource Aware** | Includes features like automatic batch size calculation for Deep Learning models based on available GPU memory, ensuring stable training. |
| ✅ **Modular & Extensible** | The object-oriented design (`WellForecastPipeline` class) and support for custom filters make it easy to extend and adapt for new datasets or post-processing techniques. |

---

### Supported Datasets

The pipeline is pre-configured to handle several standard industry and open-source datasets.

| Dataset | Description | Target Variable |
| :--- | :--- | :--- |
| **Volve** | Offshore oil production data | `BORE_OIL_VOL` |
| **UNISIM** | Reservoir simulation output | `QOOB` |
| **OPSD** | European energy generation | `GB_GBN_<type>_generation`|