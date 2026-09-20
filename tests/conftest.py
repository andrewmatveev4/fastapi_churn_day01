import pandas as pd
import pytest


@pytest.fixture
def synthetic_churn_df():
    data = {
        "monthly_fee": [50.0, 60.0, 45.0, 70.0, 55.0, 65.0, 40.0, 80.0],
        "usage_hours": [10.0, 5.0, 20.0, 3.0, 15.0, 7.0, 25.0, 2.0],
        "support_requests": [1, 3, 0, 5, 2, 4, 0, 6],
        "account_age_months": [12, 3, 24, 1, 18, 6, 36, 2],
        "failed_payments": [0, 2, 0, 3, 1, 2, 0, 4],
        "autopay_enabled": [1, 0, 1, 0, 1, 0, 1, 0],
        "region": ["america", "europe", "asia", "america", "europe", "asia", "america", "europe"],
        "device_type": [
            "desktop", "mobile", "desktop", "mobile", "desktop", "mobile", "desktop", "mobile"
            ],
        "payment_method": ["card", "paypal", "card", "paypal", "card", "paypal", "card", "paypal"],
        "churn": [0, 1, 0, 1, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def reset_model_state():
    from main import model_state
    model_state["bundle"] = None
    yield
    model_state["bundle"] = None
