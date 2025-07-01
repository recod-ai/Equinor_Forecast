# Equinor_Forecast

**An Framework for Energy-Sector Time-Series Prediction**

Equinor_Forecast is an open-source pipeline that delivers robust medium-term cumulative forecasts (8–16 weeks) for oil, gas, and renewables. It is designed for industrial settings where historical data is often sparse and operating conditions change. The framework excels at few-shot learning, continuous adaptation, and benchmarking a wide range of forecasting models.

## 🌐 2. Two Forecasting Flows

### **Flow 1 — Online Few-Shot Training**

**Use Case:** Low-latency updates, quick adaptation

📈 Uses a rolling window of 150 samples
🔁 Online fine-tuning with last 1–2 samples
🔄 Daily re-evaluation with learning rate scheduling
🧠 Compatible with XGBoost, Custom DL, ARIMA, N-Beats, etc.


## Key Features

*   **🎯 Cumulative Forecasting:** Predicts aggregated sums (e.g., total production over 8 weeks) to smooth noise and aid strategic planning.
*   **🧩 Few-Shot Learning:** Trains effectively on as few as **150 historical samples**.
*   **⚙️ Two Core Workflows:**
    1.  **Online Learning:** Continuously fine-tunes models like XGBoost, N-Beats, and NHiTS on a rolling window of recent data.
    2.  **Physics-Informed Neural Networks (PINNs):** A hybrid approach that combines deep learning with physical principles (e.g., Arps decline curves) in a single batch-trained model.
*   **📊 Comprehensive Benchmarking:** Evaluates modern (N-Beats, NHiTS, TiDE) and classic (ARIMA) models on public energy datasets.
*   **🔬 Built-in Interpretability:** Tracks feature importance over time using a longitudinal SHAP analysis.

![Overview](Overview.png)

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/recod-ai/Equinor_Forecast.gitc
cd Equinor_Forecast
pip install -r requirements.txt

### 2. Model Portfolio
* Category	Representative Models	Training Mode
* PINN Hybrid	Seq2Context (CNN + LSTM + Physics)	Batch
* Few-shot Online	XGBoost, Custom DL Architectures	Online
* Deep Learning	N-Beats, NHiTS, TiDE, TiDE+RIN, NLinear	Online
* Statistical	ARIMA, AutoARIMA, Linear Regression	Online
* Evaluation & Interpretability

