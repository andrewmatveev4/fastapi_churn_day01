import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


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


def build_preprocessor(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )
    return preprocessor


def build_model_pipeline(numeric_features, categorical_features):
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    return model


def train_churn_model():
    X, y, numeric_features, categorical_features = prepare_data()
    if len(X) == 0:
        raise ValueError("Dataset is empty")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    model = build_model_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return {
        "accuracy": accuracy,
        "f1": f1,
    }
