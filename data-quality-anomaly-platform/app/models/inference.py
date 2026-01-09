import pandas as pd
import mlflow.sklearn

MODEL_NAME = "data_quality_anomaly_model"
MODEL_STAGE = "Production"

def load_model():
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    return mlflow.sklearn.load_model(model_uri)

def score_anomaly(model, features: dict):
    X = pd.DataFrame([features])

    score = model.decision_function(X)[0]
    prediction = model.predict(X)[0]

    severity = "high" if prediction == -1 else "normal"

    return {
        "anomaly_score": float(score),
        "prediction": "anomaly" if prediction == -1 else "normal",
        "severity": severity
    }
