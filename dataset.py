import pandas as pd


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