import pandas as pd
from app.services.data_quality import run_data_quality_checks
from app.services.feature_engineering import generate_features
from app.models.inference import score_anomaly


class DummyModel:
    """
    Mock model for testing inference logic
    (avoids loading MLflow model in unit tests)
    """
    def decision_function(self, X):
        return [-0.5]  # simulate anomaly score

    def predict(self, X):
        return [-1]    # simulate anomaly


def test_structured_data_pipeline():
    # ----------------------------
    # Step 1: Create sample data
    # ----------------------------
    data = {
        "amount": [100, 120, 130, None, 5000],   # anomaly spike
        "quantity": [1, 1, 2, 1, 50]
    }
    df = pd.DataFrame(data)

    # ----------------------------
    # Step 2: Data Quality Checks
    # ----------------------------
    quality_result = run_data_quality_checks(df)

    assert quality_result["data_type"] == "structured"
    assert "missing_values" in quality_result["quality_report"]
    assert quality_result["quality_report"]["missing_values"]["amount"] == 1

    # ----------------------------
    # Step 3: Feature Engineering
    # ----------------------------
    features = generate_features(df, quality_result["data_type"])

    assert "amount_mean" in features
    assert "row_count" in features
    assert features["row_count"] == 5

    # ----------------------------
    # Step 4: Anomaly Detection
    # ----------------------------
    model = DummyModel()
    anomaly_result = score_anomaly(model, features)

    assert anomaly_result["prediction"] == "anomaly"
    assert anomaly_result["severity"] == "high"


def test_unstructured_text_pipeline():
    # ----------------------------
    # Step 1: Create text data
    # ----------------------------
    df = pd.DataFrame({
        "text": [
            "System started",
            "User logged in",
            None,
            "ERROR: connection timeout occurred",
            "Shutdown initiated"
        ]
    })

    # ----------------------------
    # Step 2: Data Quality Checks
    # ----------------------------
    quality_result = run_data_quality_checks(df)

    assert quality_result["data_type"] == "unstructured"
    assert "avg_text_length" in quality_result["quality_report"]

    # ----------------------------
    # Step 3: Feature Engineering
    # ----------------------------
    features = generate_features(df, quality_result["data_type"])

    assert "avg_text_length" in features
    assert "line_count" in features
    assert features["line_count"] == 5
