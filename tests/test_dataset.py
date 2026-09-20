from dataset import prepare_data, train_churn_model


def test_prepare_data_splits_x_and_y():
    X, y, numeric_features, categorical_features = prepare_data()

    assert "churn" not in X.columns
    assert len(numeric_features) == 6
    assert len(categorical_features) == 3
    assert len(X) == len(y)


def test_train_churn_model_produces_metrics(monkeypatch, synthetic_churn_df):
    import dataset
    monkeypatch.setattr(dataset, "load_dataset", lambda: synthetic_churn_df)

    bundle = train_churn_model("logreg", {})

    assert "accuracy" in bundle["metrics"]
    assert "f1" in bundle["metrics"]
    assert "roc_auc" in bundle["metrics"]
    assert 0.0 <= bundle["metrics"]["accuracy"] <= 1.0
    assert bundle["model_type"] == "logreg"
