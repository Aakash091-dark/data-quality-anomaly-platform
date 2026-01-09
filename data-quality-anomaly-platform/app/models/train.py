import pandas as pd
import mlflow
import mlflow.sklearn

from ml.anomaly.isolation_forest import build_model

MODEL_NAME = "data_quality_anomaly_model"

def train_anomaly_model(feature_records):
    X = pd.DataFrame(feature_records)
    model = build_model()

    with mlflow.start_run() as run:
        model.fit(X)

        mlflow.sklearn.log_model(model, "anomaly_model")
        mlflow.log_metric("training_samples", X.shape[0])
        mlflow.log_metric("feature_count", X.shape[1])

        mlflow.register_model(
            f"runs:/{run.info.run_id}/anomaly_model",
            MODEL_NAME
        )

    return model
