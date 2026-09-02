from fastapi import FastAPI
from pydantic import BaseModel
from dataset import get_preview, get_info, split_data

app = FastAPI()

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

