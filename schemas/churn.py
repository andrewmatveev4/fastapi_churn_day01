from pydantic import BaseModel


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


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict = {}
