from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------
# Production T-Learner
# ---------------------------------------------------------

from uplift.t_learner import TLearner


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------

CONTROL_MODEL_PATH = (
    MODELS_DIR / "t_learner_control_model.joblib"
)

TREATMENT_MODEL_PATH = (
    MODELS_DIR / "t_learner_treatment_model.joblib"
)

MODEL_NAME = "t_learner"
MODEL_VERSION = "1.0"


# ---------------------------------------------------------
# Load trained models
# ---------------------------------------------------------

control_model = joblib.load(
    CONTROL_MODEL_PATH
)

treatment_model = joblib.load(
    TREATMENT_MODEL_PATH
)


uplift_model = TLearner(
    control_model=control_model,
    treatment_model=treatment_model,
    clone_models=False
)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Customer Uplift Modeling API",
    description=(
        "Production API for individual treatment effect "
        "prediction using a T-Learner."
    ),
    version=MODEL_VERSION,
)


# ---------------------------------------------------------
# Input schema
# ---------------------------------------------------------

class UpliftRequest(BaseModel):

    f0: float
    f1: float
    f2: float
    f3: float
    f4: float
    f5: float
    f6: float
    f7: float
    f8: float
    f9: float
    f10: float
    f11: float


# ---------------------------------------------------------
# Output schema
# ---------------------------------------------------------

class UpliftResponse(BaseModel):

    model_config = {
        "protected_namespaces": ()
    }

    predicted_uplift: float
    recommendation: str
    model_name: str
    model_version: str


# ---------------------------------------------------------
# Treatment policy
# ---------------------------------------------------------

def get_treatment_recommendation(
    uplift: float
) -> str:

    if uplift > 0:
        return "TREAT"

    return "DO_NOT_TREAT"


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=UpliftResponse
)
def predict_uplift(
    request: UpliftRequest
):

    input_data = pd.DataFrame(
        [
            {
                "f0": request.f0,
                "f1": request.f1,
                "f2": request.f2,
                "f3": request.f3,
                "f4": request.f4,
                "f5": request.f5,
                "f6": request.f6,
                "f7": request.f7,
                "f8": request.f8,
                "f9": request.f9,
                "f10": request.f10,
                "f11": request.f11,
            }
        ]
    )

    uplift_prediction = uplift_model.predict_uplift(
        input_data
    )

    uplift = float(
        np.asarray(
            uplift_prediction
        ).reshape(-1)[0]
    )

    recommendation = (
        get_treatment_recommendation(
            uplift
        )
    )

    return UpliftResponse(
        predicted_uplift=uplift,
        recommendation=recommendation,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )