from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dataset import get_preview, get_info, split_data, train_churn_model
from contextlib import asynccontextmanager
from model_store import load_churn_model
import pandas as pd
from typing import Union

model_state = {"bundle": None}

TYPE_MAP = {
    "number": "float",
    "integer": "int",
    "string": "str",
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state["bundle"] = load_churn_model()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}


class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


class PredictionResponseChurn(BaseModel):
    predicted_class: int
    probabilities: list[float]


class TrainingConfigChurn(BaseModel):
    model_type: str
    hyperparameters: dict = {}


@app.post("/predict")
def predict(payload: Union[FeatureVectorChurn, list[FeatureVectorChurn]]):
    if model_state["bundle"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not trained yet. Call POST /model/train first.",
        )
    if isinstance(payload, list):
        clients = payload
    else:
        clients = [payload]

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


@app.get("/dataset/preview")
def dataset_preview(n: int = 10):
    return get_preview(n)


@app.get("/dataset/info")
def dataset_info():
    return get_info()


@app.get("/dataset/split-info")
def dataset_split_info():
    return split_data()


@app.post("/model/train")
def model_train(config: TrainingConfigChurn):
    try:
        bundle = train_churn_model(config.model_type, config.hyperparameters)
        model_state["bundle"] = bundle
        return {
            "metrics": bundle["metrics"],
            "trained_at": bundle["trained_at"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model/status")
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


@app.get("/model/schema")
def model_schema():
    schema = FeatureVectorChurn.model_json_schema()
    features = {}
    for name, info in schema["properties"].items():
        json_type = info["type"]
        features[name] = TYPE_MAP[json_type]
    return {"features": features}