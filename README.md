# ✈️ Airline Route Optimization & AI Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange.svg)
![Airflow](https://img.shields.io/badge/Airflow-2.7+-teal.svg)
![RAG](https://img.shields.io/badge/RAG-Powered-purple.svg)

A comprehensive flight delay prediction and optimization system featuring an AI-powered assistant built with RAG (Retrieval-Augmented Generation) architecture.

---

## 🎯 Project Overview

This project analyzes **4.5+ million US domestic flights** to predict delays, optimize routes, and provide intelligent recommendations through a conversational AI assistant.

### Key Highlights

- **ML Model**: 82.67% accuracy using LightGBM ensemble
- **Real-Time Data**: Live flight status via AviationStack API
- **AI Assistant**: RAG-powered chatbot with Groq/Llama 3.3
- **Vector Search**: ChromaDB for semantic retrieval
- **Automation**: Airflow DAGs for daily pipeline updates
- **Cost Analysis**: $47M+ annual delay costs quantified

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                               │
│                  (Streamlit Dashboard)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│   │   Overview  │  │   Delay     │  │      AI Assistant       │ │
│   │  Dashboard  │  │  Predictor  │  │    (RAG + Groq LLM)     │ │
│   └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                      RAG PIPELINE                                │
│                                                                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│   │  Chunking   │──▶  Embedding  │──▶   ChromaDB Vector DB    │ │
│   │  Pipeline   │  │  (MiniLM)   │  │    (41+ documents)      │ │
│   └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                     DATA SOURCES                                 │
│                                                                  │
│   ┌─────────────────────┐      ┌─────────────────────────────┐  │
│   │   Historical Data   │      │     Real-Time API           │  │
│   │   (2019 - 4.5M)     │      │    (AviationStack)          │  │
│   └─────────────────────┘      └─────────────────────────────┘  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                     AUTOMATION                                   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Apache Airflow DAGs                         │   │
│   │     (Daily data fetch, embedding updates, QA checks)     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Phase 1: ML & Analytics
- 📊 Exploratory Data Analysis on 4.5M+ flights
- 🤖 LightGBM model with 82.67% accuracy
- 💰 Cost simulator quantifying delay impact
- 📈 Interactive Streamlit dashboard

### Phase 2: RAG & AI Assistant
- 🧠 RAG-powered conversational AI
- 🔍 Semantic search with ChromaDB
- ⚡ Real-time flight data integration
- 🔄 Automated Airflow pipelines
- 💬 Natural language Q&A interface

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **ML/AI** | LightGBM, XGBoost, Scikit-learn |
| **RAG** | ChromaDB, Sentence-Transformers |
| **LLM** | Groq API (Llama 3.3 70B) |
| **Data** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Orchestration** | Apache Airflow, Docker |
| **APIs** | AviationStack (real-time flights) |

---

## 📁 Project Structure
```
airline-route-optimization/
│
├── app/                             # Streamlit Dashboard
│   ├── app.py                       # Main dashboard
│   └── pages/
│       └── 05_AI_Assistant.py       # AI Chatbot
│
├── rag/                             # RAG System
│   ├── chunking/
│   │   └── text_chunker.py          # Document chunking
│   ├── vectorstore/
│   │   └── chroma_store.py          # ChromaDB operations
│   └── pipeline.py                  # End-to-end RAG
│
├── knowledge_base/                  # Knowledge Documents
│   ├── historical/                  # 2019 flight analysis
│   │   ├── 01_overview.md
│   │   ├── 02_airlines.md
│   │   ├── 03_airports.md
│   │   ├── 04_time_patterns.md
│   │   └── 05_cost_analysis.md
│   └── current/                     # Real-time status
│       └── today_status.md
│
├── airflow/                         # Pipeline Automation
│   ├── dags/
│   │   ├── daily_flight_pipeline.py
│   │   └── rag_quality_check.py
│   └── docker-compose.yaml
│
├── data/
│   └── api/
│       └── aviation_stack.py        # Real-time API wrapper
│
├── notebooks/                       # Analysis Notebooks
│   ├── 01_EDA.ipynb
│   └── 02_ML_Model.ipynb
│
├── config/
│   └── settings.py                  # Configuration
│
├── models/                          # Trained Models
│   └── lgb_model.pkl
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (for Airflow)

### Installation
```bash
# Clone repository
git clone https://github.com/tirth80/airline-route-optimization.git
cd airline-route-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create `.env` file:
```bash
AVIATIONSTACK_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

### Run the Application
```bash
# Start AI Assistant
PYTHONPATH=. streamlit run app/pages/05_AI_Assistant.py

# Start Airflow (optional)
cd airflow
docker-compose up -d
```

---

## 💬 AI Assistant Demo

Ask questions like:

| Question | Type |
|----------|------|
| "What is today's flight status at JFK?" | Real-time |
| "Which airline has the best on-time performance?" | Historical |
| "Should I fly from ATL or ORD today?" | Recommendation |
| "Compare Delta and United Airlines" | Comparison |
| "Give me tips for avoiding flight delays" | Advice |

### Sample Interaction
```
User: What is the current delay status at all airports?

AI: According to today's data, the current delay status is:
    1. ATL: 8% delay rate - Excellent
    2. JFK: 12% delay rate - Good  
    3. LAX: 18% delay rate - Moderate
    4. ORD: 25% delay rate - High Delays
    
    ATL is performing best, while ORD has weather-related delays.
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 82.67% |
| ROC-AUC | 0.7496 |
| Precision | 79.3% |
| Recall | 74.1% |

---

## 📈 Key Insights

| Category | Best | Worst |
|----------|------|-------|
| **Time of Day** | 5-7 AM (12% delays) | 6-9 PM (28% delays) |
| **Day of Week** | Saturday (15%) | Thursday (22%) |
| **Month** | September (14%) | June (24%) |
| **Airline** | Delta (82% on-time) | Frontier (72% on-time) |

### Cost Impact
- **Total Annual Delay Cost**: $47M+ at major hubs
- **Average Delay Duration**: 57 minutes
- **Cost per 2-hour Delay**: ~$7,800 per flight

---

## 🔄 Airflow DAGs

| DAG | Schedule | Purpose |
|-----|----------|---------|
| `daily_flight_pipeline` | Daily 2AM | Fetch API data, update knowledge base, refresh embeddings |
| `rag_quality_check` | Daily 6AM | Test retrieval, validate pipeline, generate health report |

Access Airflow UI: http://localhost:8080 (admin/admin)

---

## 🎯 Future Enhancements

- [ ] Weather API integration for better predictions
- [ ] Flight price optimization module
- [ ] Mobile app version
- [ ] Multi-language support
- [ ] Voice interface

---

## 👤 Author

**Tirth Patel**

[![GitHub](https://img.shields.io/badge/GitHub-tirth80-black?logo=github)](https://github.com/tirth80)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
```

---

