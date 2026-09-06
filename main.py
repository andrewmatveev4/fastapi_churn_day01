from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dataset import get_preview, get_info, split_data, train_churn_model
from contextlib import asynccontextmanager
from model_store import load_churn_model
import pandas as pd

model_state = {"bundle": None}

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


@app.post("/predict", response_model=list[PredictionResponseChurn])
def predict(clients: list[FeatureVectorChurn]):
    if model_state["bundle"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not trained yet. Call POST /model/train first.",
        )
    model = model_state["bundle"]["model"]
    df = pd.DataFrame([c.model_dump() for c in clients])
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
    return results


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
def model_train():
    try:
        bundle = train_churn_model()
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
    }