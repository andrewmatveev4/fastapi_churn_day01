import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset():
    df = pd.read_csv("data/churn_dataset.csv")
    return df


def get_preview(n):
    df = load_dataset()
    preview = df.head(n)
    return preview.to_dict(orient="records")


def get_info():
    df = load_dataset()
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "features": list(df.columns),
        "churn_distribution": df["churn"].value_counts().to_dict(),
    }


def prepare_data():
    df = load_dataset()
    df = df.dropna()
    X = df.drop(columns=["churn"])
    y = df["churn"]
    numeric_features = [
        "monthly_fee",
        "usage_hours",
        "support_requests",
        "account_age_months",
        "failed_payments",
        "autopay_enabled",
    ]
    categorical_features = [
        "region",
        "device_type",
        "payment_method",
    ]

    return X, y, numeric_features, categorical_features

def split_data():
    X, y, numeric_features, categorical_features = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_churn_distribution": y_train.value_counts().to_dict(),
        "test_churn_distribution": y_test.value_counts().to_dict(),
    }