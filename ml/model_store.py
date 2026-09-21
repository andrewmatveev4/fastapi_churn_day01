import os
import joblib

MODEL_PATH = "models/churn_model.joblib"


def save_churn_model(bundle, path=MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    joblib.dump(bundle, path)


def load_churn_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)
