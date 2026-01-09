from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from app.services.ingestion import parse_uploaded_file
from app.services.data_quality import run_data_quality_checks
from app.services.feature_engineering import generate_features
from app.models.inference import load_model, score_anomaly
from app.services.explainability import generate_explanation
from app.services.drift_service import detect_drift
from app.core.security import create_access_token
from app.api.deps import get_current_user

router = APIRouter(prefix="/api", tags=["Ingestion"])

@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Mock authentication - accept any username/password
    # In real app, verify against database
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

model = None

@router.post("/upload")
async def upload_data(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
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

@router.post("/drift")
async def check_drift(
    reference_file: UploadFile = File(...), 
    current_file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    ref_content = await reference_file.read()
    curr_content = await current_file.read()

    ref_df = parse_uploaded_file(reference_file.filename, ref_content)
    curr_df = parse_uploaded_file(current_file.filename, curr_content)

    drift_report = detect_drift(ref_df, curr_df)

    return {
        "status": "success",
        "reference_file": reference_file.filename,
        "current_file": current_file.filename,
        "drift_report": drift_report
    }
