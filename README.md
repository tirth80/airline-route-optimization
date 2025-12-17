# ✈️ Airline Route Optimization & Delay Prediction System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-82.67%25-success)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive end-to-end flight delay analysis and prediction system built using **4.5M+ US domestic flight records**.  
Features ML-powered delay prediction, cost impact simulation, and an interactive Streamlit dashboard.

> **Why this exists**  
> Flight delays cost airlines billions annually and frustrate millions of passengers.  
> This project provides actionable insights for route optimization, delay prediction, and cost quantification.

---

## 📂 Table of Contents

- [Architecture](#architecture)
- [Key Findings](#key-findings)
- [Quickstart](#quickstart)
- [Data Overview](#data-overview)
- [Usage Guide](#usage-guide)
- [Project Layout](#project-layout)
- [Modeling Details](#modeling-details)
- [Dashboard Features](#dashboard-features)
- [Cost Simulator](#cost-simulator)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

---

## 🏗 Architecture
```
CSV (4.5M+ flights)
    │
    ▼
[Data Cleaning] → Feature Engineering (29 features)
    │
    ▼
[EDA] → Visualizations (15+ charts) → Key Insights
    │
    ▼
[ML Pipeline]
    ├── Logistic Regression (Baseline)
    ├── Random Forest
    ├── XGBoost
    ├── LightGBM
    └── Gradient Boosting
    │
    ▼
[Ensemble Model] → Threshold Optimization → 82.67% Accuracy
    │
    ▼
[Streamlit Dashboard]
    ├── Overview (KPIs & Charts)
    ├── Delay Predictor (ML-powered)
    ├── Cost Simulator (Financial impact)
    └── Route Analyzer (Airport analysis)
```

---

**Design Choices**

- **Large-scale data** — 4.5M+ flight records for robust analysis
- **Ensemble approach** — Combines XGBoost + LightGBM + Random Forest
- **Threshold optimization** — Tuned for maximum accuracy (0.75 threshold)
- **Interactive UI** — Streamlit dashboard for easy exploration
- **Cost quantification** — Real-world financial impact calculations

---

## 📊 Key Findings

| Insight | Finding |
|---------|---------|
| Total Flights Analyzed | 4,542,343 |
| Overall Delay Rate | 18.9% |
| Delayed Flights | 859,158 |
| Best Month to Fly | September (14% delays) |
| Worst Month to Fly | June (24% delays) |
| Best Time to Fly | Early morning (5-7 AM) |
| Worst Time to Fly | Evening (6-9 PM) |
| Best Day to Fly | Saturday |
| Worst Day to Fly | Thursday/Friday |

---

## 🚀 Quickstart
```bash
# 1. Clone the repository
git clone https://github.com/tirth80/airline-route-optimization.git
cd airline-route-optimization

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

---

## 📁 Data Overview

### Input Features

| Feature | Description |
|---------|-------------|
| `MONTH` | Month of flight (1-12) |
| `DAY_OF_WEEK` | Day of week (1-7) |
| `DEP_TIME_BLK` | Departure time block |
| `CARRIER_NAME` | Airline name |
| `DEPARTING_AIRPORT` | Origin airport |
| `DISTANCE_GROUP` | Flight distance category |
| `CONCURRENT_FLIGHTS` | Number of concurrent flights |
| `PRCP`, `SNOW`, `TMAX`, `AWND` | Weather conditions |
| `CARRIER_HISTORICAL` | Historical carrier delay rate |
| `DEP_AIRPORT_HIST` | Historical airport delay rate |

### Target Variable

| Field | Description |
|-------|-------------|
| `DEP_DEL15` | Binary (1 = Delayed >15 min, 0 = On-time) |

---

## 📖 Usage Guide

### Streamlit Dashboard
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Jupyter Notebooks
```bash
jupyter notebook
```

- `01_data_loading_and_exploration.ipynb` — EDA & visualizations
- `02_ML_Model.ipynb` — Model training & evaluation

---

## 📂 Project Layout
```
airline-route-optimization/
│
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
│
├── data/
│   └── raw/
│       └── flights_2019.csv    # Flight data (4.5M records)
│
├── notebooks/
│   ├── 01_data_loading_and_exploration.ipynb
│   ├── 02_ML_Model.ipynb
│   └── 03_Simulator.ipynb
│
├── src/
│   ├── xgb_model.pkl           # Trained XGBoost model
│   ├── lgb_model.pkl           # Trained LightGBM model
│   ├── rf_model.pkl            # Trained Random Forest model
│   ├── scaler.pkl              # Feature scaler
│   ├── features.pkl            # Feature list
│   └── simulator.py            # Cost simulator functions
│
└── reports/
    └── visualizations/
        ├── model_comparison.png
        ├── feature_importance.png
        ├── confusion_matrix.png
        └── roc_curves.png
```

---

## 🤖 Modeling Details

### Models Compared

| Model | Accuracy | ROC AUC |
|-------|----------|---------|
| Logistic Regression | 62.91% | 0.6721 |
| Random Forest | 77.10% | 0.7161 |
| XGBoost | 72.37% | 0.7183 |
| LightGBM | 67.06% | 0.7197 |
| Gradient Boosting | 81.43% | 0.6985 |
| **Ensemble (Tuned)** | **82.67%** | **0.7496** |

### Final Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 82.67% |
| **ROC AUC** | 0.7496 |
| **Precision** | 68.40% |
| **Recall** | 15.60% |
| **F1 Score** | 25.41% |
| **Optimal Threshold** | 0.75 |

### Top Predictive Features

1. `DEP_BLOCK_HIST` — Historical departure block delay rate
2. `PRCP` — Precipitation
3. `CARRIER_HISTORICAL` — Carrier's historical delay rate
4. `AVG_MONTHLY_PASS_AIRLINE` — Average monthly passengers
5. `DEP_TIME_BLK` — Departure time block

---

## 🖥️ Dashboard Features

### 1. Overview Dashboard
- Total flights, delay rate, KPIs
- Monthly delay trends
- Day of week analysis
- Airline performance comparison

### 2. Delay Predictor
- Select month, day, airline, airport
- ML-powered delay probability
- Risk level assessment (Low/Medium/High)
- Historical comparison

### 3. Cost Simulator
- Input delay duration & passengers
- Calculates fuel, crew, passenger costs
- Customer satisfaction (NPS) impact
- Annual cost projections

### 4. Route Analyzer
- Airport-level performance analysis
- Best/worst months and carriers
- Delay trends over time
- Estimated annual delay costs

---

## 💰 Cost Simulator

### Cost Breakdown

| Cost Component | Rate |
|----------------|------|
| Fuel | $40/minute |
| Crew | $25/minute |
| Maintenance | $15/minute |
| Passenger Compensation (>1hr) | $10/passenger |
| Passenger Compensation (>2hr) | $25/passenger |
| Passenger Compensation (>3hr) | $75/passenger |

### Example Calculation
```
60-minute delay with 150 passengers:
────────────────────────────────────
  Fuel:         $2,400
  Crew:         $1,500
  Maintenance:    $900
  Passengers:   $1,500
  ─────────────────────
  TOTAL:        $6,300
  NPS Impact:   -10 points
```

---

## 📈 Results

### Business Insights

- **$47M+** estimated annual delay cost at major hubs
- **16%** improvement possible by shifting flights to morning
- **September** identified as optimal month for travel
- **Thursday/Friday evenings** are highest risk periods

### What-If Scenarios

| Scenario | Current | Proposed | Savings |
|----------|---------|----------|---------|
| Shift evening → morning | 28% delays | 12% delays | $3.2M/year |
| Reduce congestion | 32% delays | 15% delays | $5.1M/year |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.10+ |
| **Data** | Pandas, NumPy |
| **ML** | Scikit-learn, XGBoost, LightGBM |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Notebook** | Jupyter |

---

## 🚀 Future Enhancements

- [ ] Real-time weather API integration
- [ ] Deploy to Streamlit Cloud
- [ ] Add arrival delay prediction
- [ ] Include more airports/carriers
- [ ] Build RAG-powered AI assistant (Phase 2)
- [ ] Mobile-responsive dashboard

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact

**Tirth Patel**  
GitHub: [@tirth80](https://github.com/tirth80)

---

## 📄 License

This project is licensed under the MIT License.

---

*Data Source: Kaggle - 2019 Airline Delays and Cancellations*
