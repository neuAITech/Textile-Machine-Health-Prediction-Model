# 🏭 Monitoring Yarn Machine Health and Efficiency

> End-to-end ML pipeline & interactive Streamlit dashboard for monitoring yarn machine health and efficiency — tracks health, predicts remaining useful life (RUL), and provides actionable maintenance recommendations.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-Enabled-006600?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LightGBM-Enabled-9B59B6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Multi-Target ML Pipeline** | Trains & evaluates 6 algorithms across 4 prediction targets |
| 📊 **Interactive Dashboard** | Real-time KPIs, sensor gauges, fleet health overview |
| 🔮 **Prediction Engine** | Instant health & efficiency predictions with quick scenario presets |
| 📈 **Model Analytics** | Compare model performance with interactive charts |
| 🔍 **Data Explorer** | Browse, filter & correlate the raw dataset |
| 📋 **Automated EDA** | Generate comprehensive EDA reports in one command |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML / Data** | Python · NumPy · Pandas · Scikit-learn · XGBoost · LightGBM |
| **Visualization** | Plotly · Matplotlib · Seaborn |
| **Dashboard** | Streamlit |
| **Serialization** | Joblib |

---

## 📁 Project Structure

```
Monitoring_Yarn_Machine_Health/
│
├── ml_pipeline.py            # Complete ML training & evaluation pipeline
├── streamlit_app.py          # Streamlit dashboard (4 pages)
├── eda_analysis.py           # Exploratory data analysis script
├── data.csv                  # Source dataset (5,001 rows × 30 columns)
├── requirements.txt          # Python dependencies
├── README.md
│
├── models/                   # Trained model artifacts
│   ├── Failure_Mode_Code_best_model.pkl
│   ├── Failure_Imminent_Flag_best_model.pkl
│   ├── Target_Failure_24H_best_model.pkl
│   ├── Target_RUL_Hours_best_model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
│
├── results/                  # Training results & predictions
│   ├── final_summary.csv
│   ├── test_predictions.csv
│   └── *_results.csv
│
└── eda_results/              # EDA output (plots & summaries)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+

### Installation & Usage

```bash
# 1. Clone the repo
git clone https://github.com/AjAyrAo43/Machinery_Health_Predication.git
cd Machinery_Health_Predication

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the models (required before dashboard)
python ml_pipeline.py

# 4. Launch the dashboard
streamlit run streamlit_app.py

# 5. (Optional) Run EDA
python eda_analysis.py
```

---

## 🧠 ML Pipeline

### Prediction Targets

| # | Target | Type | Description |
|---|--------|------|-------------|
| 1 | `Target_RUL_Hours` | Regression | Remaining Useful Life in hours |
| 2 | `Failure_Imminent_Flag` | Binary | 1 = failure likely soon (RUL < 1,250 h) |
| 3 | `Target_Failure_24H` | Binary | 1 = failure within critical zone (RUL ≤ 1,100 h) |
| 4 | `Failure_Mode_Code` | Multi-class | 0-Healthy · 1-Early Wear · 2-Moderate Wear · 3-Critical |

### Models

| Algorithm | Classification | Regression |
|-----------|:-:|:-:|
| Random Forest | ✅ | ✅ |
| Gradient Boosting | ✅ | ✅ |
| XGBoost | ✅ | ✅ |
| LightGBM | ✅ | ✅ |
| Extra Trees | ✅ | ✅ |
| Logistic / Ridge Regression | ✅ | ✅ |

### Pipeline Workflow

```
Data Loading → Feature Engineering → Scaling & Splitting → Model Training (5-fold CV) → Evaluation → Save Best Models
```

**Evaluation Metrics:**
- **Classification** → Accuracy, Precision, Recall, F1-Score
- **Regression** → MAE, RMSE, R²

---

## 📊 Dashboard Pages

### 1️⃣ Dashboard / Overview
Fleet-wide KPIs — machine count, average RUL, efficiency, critical rate, energy consumption, and total output. Includes degradation distribution, RUL histogram, scatter plots, and live sensor gauges.

### 2️⃣ Prediction Engine
Enter machine parameters manually or use **quick scenario presets** (Normal / Failure) to get predictions across all 4 targets. Color-coded health banner (Healthy / Needs Attention / Critical) with an interactive RUL gauge.

### 3️⃣ Model Analytics
Compare all trained models with interactive bar charts, metrics tables, and a final summary of best-performing models per target.

### 4️⃣ Data Explorer
Browse & filter the raw dataset, view statistical summaries, and create custom correlation analyses with interactive Plotly charts.

---

## 📦 Dataset

**5,001 records** × **30 features** covering:

| Category | Features |
|----------|----------|
| **Machine Identity** | Machine ID, Type, Section, State, Shift |
| **Sensor Readings** | Speed (RPM), Temperature, Humidity, Motor Current, Voltage Variation |
| **Production Metrics** | Output (kg), Waste (kg), Yarn Breaks, Energy (kWh) |
| **Health Indicators** | Wear Score, Friction Index, Efficiency Index, Component Health %, RUL Hours |
| **Safety Flags** | Safety Interlock, Auto Shutdown, Emergency Stop Count |

---
