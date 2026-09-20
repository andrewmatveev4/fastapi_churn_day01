from fastapi.testclient import TestClient
from main import app


def test_predict_without_model_returns_503():
    client = TestClient(app)

    response = client.post("/predict", json={
        "monthly_fee": 50.0,
        "usage_hours": 10.0,
        "support_requests": 2,
        "account_age_months": 12,
        "failed_payments": 0,
        "region": "europe",
        "device_type": "mobile",
        "payment_method": "card",
        "autopay_enabled": 1,
    })

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "MODEL_NOT_TRAINED"


def test_train_status_predict_flow(monkeypatch, synthetic_churn_df):
    import dataset
    monkeypatch.setattr(dataset, "load_dataset", lambda: synthetic_churn_df)

    client = TestClient(app)

    train_resp = client.post("/model/train", json={
        "model_type": "logreg",
        "hyperparameters": {"max_iter": 1000},
    })
    assert train_resp.status_code == 200
    assert "metrics" in train_resp.json()

    status_resp = client.get("/model/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["trained"] is True

    predict_resp = client.post("/predict", json={
        "monthly_fee": 50.0,
        "usage_hours": 10.0,
        "support_requests": 2,
        "account_age_months": 12,
        "failed_payments": 0,
        "region": "europe",
        "device_type": "mobile",
        "payment_method": "card",
        "autopay_enabled": 1,
    })
    assert predict_resp.status_code == 200
    assert "predicted_class" in predict_resp.json()


def test_train_unknown_model_type_returns_400(monkeypatch, synthetic_churn_df):
    import dataset
    monkeypatch.setattr(dataset, "load_dataset", lambda: synthetic_churn_df)

    client = TestClient(app)

    train_resp = client.post("/model/train", json={
        "model_type": "unknown_model",
        "hyperparameters": {},
    })
    assert train_resp.status_code == 400
    body = train_resp.json()
    assert body["code"] == "UNKNOWN_MODEL_TYPE"


def test_predict_invalid_input_returns_422():
    client = TestClient(app)

    response = client.post("/predict", json={
        "monthly_fee": "invalid_string",
        "usage_hours": 10.0,
        "support_requests": 2,
        "account_age_months": 12,
        "failed_payments": 0,
        "region": "europe",
        "device_type": "mobile",
        "payment_method": "card",
        "autopay_enabled": 1,
    })

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_health_returns_status():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "model_available" in body
    assert "dataset_available" in body
    assert body["dataset_available"] is True
