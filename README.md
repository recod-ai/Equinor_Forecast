# Equinor_Forecast

**An Framework for Energy-Sector Time-Series Prediction**

Equinor_Forecast is an open-source pipeline that delivers robust medium-term cumulative forecasts (8–16 weeks) for oil, gas, and renewables. It is designed for industrial settings where historical data is often sparse and operating conditions change. The framework excels at few-shot learning, continuous adaptation, and benchmarking a wide range of forecasting models.

Most industrial assets lack the long, tidy histories that conventional ML expects. Waiting years to collect data delays value creation.

## 🌐 Two Forecasting Flows

### **Flow 1 — Online Few-Shot Training**

**Use Case:** Low-latency updates, quick adaptation

📈 Uses a rolling window of 150 samples
🔁 Online fine-tuning with last 1–2 samples
🔄 Daily re-evaluation with learning rate scheduling
🧠 Compatible with XGBoost, Custom DL, ARIMA, N-Beats, etc.


![Online Training Diagram](Online.png)


### **Flow 2 — PINNs with Batch Few-Shot Training**

**Use Case:** Domain-informed long-horizon forecasts

🧪 Trained once on \~40% of data
📦 Encodes physics via exponential, Arps, pressure-based strategies
🎯 Combines deep temporal encoders with physics baselines
⚖️ Predicts production using blended residual learning


![PINNs Overview](PINN.png)


## Key Features

*   **🎯 Cumulative Forecasting:** Predicts aggregated sums (e.g., total production over 8 weeks) to smooth noise and aid strategic planning.
*   **🧩 Few-Shot Learning:** Trains effectively on as few as **150 historical samples**.
*   **📊 Comprehensive Benchmarking:** Evaluates modern (N-Beats, NHiTS, TiDE) and classic (ARIMA) models on public energy datasets.
*   **🔬 Built-in Interpretability:** Tracks feature importance over time using a longitudinal SHAP analysis.

![Overview](Overview.png)

### Datasets

| Domain              | Dataset         | Key Facts                               |
| ------------------- | --------------- | --------------------------------------- |
| Oil & gas           | **Volve**       | 4 wells, daily, target = BORE\_OIL\_VOL |
| Synthetic reservoir | **UNISIM-II-H** | 10 producers, 3 263 days, target = QOOB |
| Renewables          | **OPSD** (GB)   | Wind (30 min→1 d), Solar & Load (12 h)  |

![Pipeline Architecture](Data_Pipeline.png)


---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/recod-ai/Equinor_Forecast.gitc
cd Equinor_Forecast
pip install -r requirements.txt
```

### 2. Model Portfolio
* Category	Representative Models	Training Mode
* PINN Hybrid	Seq2Context (CNN + LSTM + Physics)	Batch
* Few-shot Online	XGBoost, Custom DL Architectures	Online
* Deep Learning	N-Beats, NHiTS, TiDE, TiDE+RIN, NLinear	Online
* Statistical	ARIMA, AutoARIMA, Linear Regression	Online
* Evaluation & Interpretability

