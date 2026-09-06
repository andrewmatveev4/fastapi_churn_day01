from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dataset import get_preview, get_info, split_data, train_churn_model
from contextlib import asynccontextmanager
from model_store import load_churn_model

model_state = {"bundle": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state["bundle"] = load_churn_model()
    yield
    # код ПОСЛЕ yield = выполняется при остановке

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


@app.post("/predict")
def predict(features: FeatureVectorChurn):
    return features


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