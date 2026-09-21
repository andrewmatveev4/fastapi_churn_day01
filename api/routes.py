import os
import logging
import pandas as pd
from fastapi import APIRouter

from core.state import model_state
from core.errors import ChurnServiceError
from schemas.churn import (
    FeatureVectorChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
    ErrorResponse,
)
from ml.pipeline import get_preview, get_info, split_data, train_churn_model
from ml.history_store import load_history
from typing import Union

logger = logging.getLogger("churn_service")

router = APIRouter()

TYPE_MAP = {
    "number": "float",
    "integer": "int",
    "string": "str",
}


@router.get("/")
def read_root():
    return {"message": "ml churn service is running"}


@router.post("/predict", responses={
    503: {
        "description": "Model not trained",
        "model": ErrorResponse,
        "content": {"application/json": {"example": {
            "code": "MODEL_NOT_TRAINED",
            "message": "Model is not trained yet. Call POST /model/train first.",
            "details": {},
        }}},
    },
    422: {
        "description": "Request validation failed",
        "model": ErrorResponse,
        "content": {"application/json": {"example": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed.",
            "details": {"errors": []},
        }}},
    },
    500: {
        "description": "Internal server error",
        "model": ErrorResponse,
        "content": {"application/json": {"example": {
            "code": "INTERNAL_ERROR",
            "message": "Internal server error.",
            "details": {},
        }}},
    },
})
def predict(payload: Union[FeatureVectorChurn, list[FeatureVectorChurn]]):
    if model_state["bundle"] is None:
        raise ChurnServiceError(
            code="MODEL_NOT_TRAINED",
            message="Model is not trained yet. Call POST /model/train first.",
            status_code=503,
        )
    if isinstance(payload, list):
        clients = payload
    else:
        clients = [payload]

    logger.info("Predict called: %s client(s)", len(clients))
    bundle = model_state["bundle"]
    model = bundle["model"]
    feature_order = bundle["numeric_features"] + bundle["categorical_features"]
    df = pd.DataFrame([c.model_dump() for c in clients])[feature_order]
    prediction = model.predict(df)
    proba = model.predict_proba(df)

    results = []
    for i in range(len(clients)):
        results.append(
            PredictionResponseChurn(
                predicted_class=int(prediction[i]),
                probabilities=proba[i].tolist(),
            )
        )
    if isinstance(payload, list):
        return results
    else:
        return results[0]


@router.get("/dataset/preview")
def dataset_preview(n: int = 10):
    return get_preview(n)


@router.get("/dataset/info")
def dataset_info():
    return get_info()


@router.get("/dataset/split-info")
def dataset_split_info():
    return split_data()


@router.post("/model/train", responses={
    400: {
        "description": "Unknown model type",
        "model": ErrorResponse,
        "content": {"application/json": {"example": {
            "code": "UNKNOWN_MODEL_TYPE",
            "message": "Unknown model_type: svm",
            "details": {"model_type": "svm", "supported": ["logreg", "random_forest"]},
        }}},
    },
    500: {
        "description": "Empty dataset",
        "model": ErrorResponse,
        "content": {"application/json": {"example": {
            "code": "EMPTY_DATASET",
            "message": "Dataset is empty, cannot train the model.",
            "details": {},
        }}},
    },
    422: {
        "description": "Request validation failed",
        "model": ErrorResponse,
        "content": {"application/json": {"example": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed.",
            "details": {"errors": []},
        }}},
    },
})
def model_train(config: TrainingConfigChurn):
    bundle = train_churn_model(config.model_type, config.hyperparameters)
    model_state["bundle"] = bundle
    return {
        "metrics": bundle["metrics"],
        "trained_at": bundle["trained_at"],
    }


@router.get("/model/status")
def model_status():
    bundle = model_state["bundle"]
    if bundle is None:
        return {"trained": False}
    return {
        "trained": True,
        "trained_at": bundle["trained_at"],
        "metrics": bundle["metrics"],
        "model_type": bundle["model_type"],
        "hyperparameters": bundle["hyperparameters"],
    }


@router.get("/model/schema")
def model_schema():
    schema = FeatureVectorChurn.model_json_schema()
    features = {}
    for name, info in schema["properties"].items():
        json_type = info["type"]
        features[name] = TYPE_MAP[json_type]
    return {"features": features}


@router.get("/model/metrics", responses={
    404: {
        "description": "No training history found",
        "model": ErrorResponse,
        "content": {"application/json": {"example": {
            "code": "NO_TRAINING_HISTORY",
            "message": "No training history yet. Train a model first.",
            "details": {},
        }}},
    },
})
def model_metrics(model_type: str | None = None):
    history = load_history()
    if len(history) == 0:
        raise ChurnServiceError(
            code="NO_TRAINING_HISTORY",
            message="No training history yet. Train a model first.",
            status_code=404,
        )
    if model_type is not None:
        history = [r for r in history if r["model_type"] == model_type]
    if len(history) == 0:
        raise ChurnServiceError(
            code="NO_TRAINING_HISTORY",
            message=f"No training history for model_type: {model_type}",
            status_code=404,
        )
    return {
        "latest": history[-1],
        "history": history,
    }


@router.get("/health")
def health():
    model_available = model_state["bundle"] is not None
    dataset_available = os.path.exists("data/churn_dataset.csv")
    status = "ok" if (model_available and dataset_available) else "degraded"
    return {
        "status": status,
        "model_available": model_available,
        "dataset_available": dataset_available,
    }
