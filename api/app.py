
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc

from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel


PROJECT_ROOT = Path(
    r"C:\Users\ugand\customer-churn-uplift-modeling"
)

MLFLOW_DIR = (
    PROJECT_ROOT / "mlruns"
)

mlflow_tracking_path = (
    MLFLOW_DIR / "mlflow.db"
)

mlflow.set_tracking_uri(
    f"sqlite:///{mlflow_tracking_path}"
)


REGISTERED_MODEL_NAME = (
    "customer_uplift_t_learner"
)

MODEL_ALIAS = "champion"

MODEL_URI = (
    f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
)


FEATURE_COLUMNS = [
    f"f{i}"
    for i in range(12)
]


model = mlflow.pyfunc.load_model(
    MODEL_URI
)


app = FastAPI(
    title="Customer Uplift Modeling API",
    description=(
        "API for uplift prediction using "
        "the MLflow champion T-Learner."
    ),
    version="1.0.0"
)


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


class UpliftResponse(BaseModel):

    predicted_uplift: float
    recommendation: str
    model_name: str
    model_alias: str


def get_treatment_recommendation(
    uplift: float
) -> str:

    if uplift > 0:
        return "TREAT"

    return "DO_NOT_TREAT"


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": REGISTERED_MODEL_NAME,
        "alias": MODEL_ALIAS
    }


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
                feature: getattr(
                    request,
                    feature
                )
                for feature in FEATURE_COLUMNS
            }
        ]
    )

    prediction = model.predict(
        input_data
    )

    uplift = float(
        np.asarray(
            prediction
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
        model_name=REGISTERED_MODEL_NAME,
        model_alias=MODEL_ALIAS
    )
