from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ingestion import parse_uploaded_file
from app.services.data_quality import run_data_quality_checks
from app.services.feature_engineering import generate_features
from app.models.inference import load_model, score_anomaly
from app.services.explainability import generate_explanation

router = APIRouter(prefix="/api", tags=["Ingestion"])

model = None

@router.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    global model

    if model is None:
        model = load_model()

    content = await file.read()
    df = parse_uploaded_file(file.filename, content)

    quality_result = run_data_quality_checks(df)
    features = generate_features(df, quality_result["data_type"])
    anomaly_result = score_anomaly(model, features)

    explanation = None
    if anomaly_result["prediction"] == "anomaly":
        explanation = generate_explanation(model, features)

    return {
        "status": "success",
        "file_name": file.filename,
        "data_type": quality_result["data_type"],
        "quality_report": quality_result["quality_report"],
        "anomaly_result": anomaly_result,
        "explanation": explanation
    }
