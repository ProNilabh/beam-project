"""
BEAM — FastAPI Web Service
==========================
Deploys the best BEAM model as a REST API.

Usage:
    cd beam-project
    uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Load model + scaler ────────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

model  = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)

# ─── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="BEAM — Building Energy Assessment with ML",
    description=(
        "Predict heating and cooling energy loads (kWh/m²) "
        "from 8 building design features."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas ─────────────────────────────────────────────────────────────────

class BuildingInput(BaseModel):
    """The 8 architectural design inputs."""
    Relative_Compactness: float = Field(
        ..., ge=0.6, le=1.0,
        description="Compactness ratio (0.62–0.98)",
    )
    Surface_Area: float = Field(
        ..., ge=500, le=900,
        description="Total surface area in m²",
    )
    Wall_Area: float = Field(
        ..., ge=200, le=450,
        description="Wall area in m²",
    )
    Roof_Area: float = Field(
        ..., ge=100, le=250,
        description="Roof area in m²",
    )
    Overall_Height: float = Field(
        ..., ge=3, le=8,
        description="Building height in metres (3.5 or 7.0)",
    )
    Orientation: float = Field(
        ..., ge=1, le=5,
        description="2=North, 3=East, 4=South, 5=West",
    )
    Glazing_Area: float = Field(
        ..., ge=0, le=1,
        description="Window-to-wall ratio (0, 0.1, 0.25, 0.4)",
    )
    Glazing_Area_Distribution: float = Field(
        ..., ge=0, le=6,
        description="0=none, 1–5=uniform to concentrated",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Relative_Compactness": 0.74,
                    "Surface_Area": 686.0,
                    "Wall_Area": 245.0,
                    "Roof_Area": 220.5,
                    "Overall_Height": 3.5,
                    "Orientation": 3.0,
                    "Glazing_Area": 0.1,
                    "Glazing_Area_Distribution": 2.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    Heating_Load_kWh_m2: float
    Cooling_Load_kWh_m2: float
    model_used: str


class BatchInput(BaseModel):
    buildings: list[BuildingInput]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def health():
    """Health-check / root."""
    return {
        "status": "healthy",
        "model": metadata["model_name"],
        "test_r2": metadata["test_r2"],
        "version": "1.0.0",
    }


@app.get("/model-info")
def model_info():
    """Return full model metadata."""
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(building: BuildingInput):
    """Predict heating & cooling loads for ONE building."""
    try:
        features = np.array([[
            building.Relative_Compactness,
            building.Surface_Area,
            building.Wall_Area,
            building.Roof_Area,
            building.Overall_Height,
            building.Orientation,
            building.Glazing_Area,
            building.Glazing_Area_Distribution,
        ]])

        scaled = scaler.transform(features)
        pred   = model.predict(scaled)

        if pred.ndim == 1:
            pred = pred.reshape(1, -1)

        return PredictionResponse(
            Heating_Load_kWh_m2=round(float(pred[0][0]), 2),
            Cooling_Load_kWh_m2=round(float(pred[0][1]), 2),
            model_used=metadata["model_name"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchInput):
    """Predict for multiple buildings."""
    try:
        features = np.array([
            [
                b.Relative_Compactness, b.Surface_Area, b.Wall_Area,
                b.Roof_Area, b.Overall_Height, b.Orientation,
                b.Glazing_Area, b.Glazing_Area_Distribution,
            ]
            for b in batch.buildings
        ])

        scaled = scaler.transform(features)
        preds  = model.predict(scaled)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 2)

        results = [
            PredictionResponse(
                Heating_Load_kWh_m2=round(float(p[0]), 2),
                Cooling_Load_kWh_m2=round(float(p[1]), 2),
                model_used=metadata["model_name"],
            )
            for p in preds
        ]
        return BatchResponse(predictions=results, count=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── run directly ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
