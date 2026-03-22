# BEAM — Building Energy Assessment with ML

> Predicting Heating & Cooling Loads Before Construction Begins

**Course:** I.MA_ARTIFIN.F2601  
**Student:** Nilabh Pandey  
**Instructor:** Forooz Shahbazi Avarvand

---

## Overview

BEAM predicts a building's **heating load** and **cooling load** (kWh/m²) from 8 architectural design features — enabling energy-efficient decisions **before construction begins**.

Buildings account for ~40 % of global energy use and ~36 % of CO₂ emissions. Most energy waste is locked in at the design stage. BEAM addresses this with instant ML-based predictions.

---

## Dataset

| Property | Value |
|----------|-------|
| Name | UCI Energy Efficiency Dataset |
| Source | [UCI ML Repository](https://archive.ics.uci.edu/dataset/242/energy+efficiency) |
| Rows | 768 |
| Missing values | 0 |
| Origin | Ecotect simulation — Tsanas & Xifara (2012) |

**8 inputs:** Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height, Orientation, Glazing Area, Glazing Area Distribution  
**2 outputs:** Heating Load (Y1), Cooling Load (Y2)

---

## Project Structure

```
beam-project/
├── data/
│   └── ENB2012_data.csv          # dataset
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # load + split + scale
│   ├── train.py                  # MLflow experiment tracking
│   └── app.py                    # FastAPI web service
├── models/                       # saved after training
│   ├── best_model.joblib
│   ├── scaler.joblib
│   └── metadata.json
├── mlruns/                       # MLflow tracking (created at runtime)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Train Models (creates 2 MLflow experiments)

```bash
cd beam-project
python -m src.train
```

This runs **7 models** across **2 experiments**:

| Experiment | Models |
|------------|--------|
| `BEAM_Experiment_1_Baselines` | Ridge, Random Forest, XGBoost |
| `BEAM_Experiment_2_Tuned` | RF (tuned), XGBoost (tuned), Gradient Boosting (tuned), Neural Net |

### 3. View MLflow UI

```bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns
```

Open **http://localhost:5000** — you will see both experiments with all runs, metrics, and model artifacts.

### 4. Start the FastAPI Service

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** for Swagger UI.

### 5. Test a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Relative_Compactness": 0.74,
    "Surface_Area": 686.0,
    "Wall_Area": 245.0,
    "Roof_Area": 220.5,
    "Overall_Height": 3.5,
    "Orientation": 3.0,
    "Glazing_Area": 0.1,
    "Glazing_Area_Distribution": 2.0
  }'
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check + model info |
| GET | `/model-info` | Full metadata |
| POST | `/predict` | Single building prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/docs` | Interactive Swagger UI |

---

## Docker (optional)

```bash
docker-compose up --build
# API  → http://localhost:8000
# MLflow → http://localhost:5000
```

---

## References

- Tsanas, A. & Xifara, A. (2012). Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools.
- Chip Huyen — *Designing Machine Learning Systems*