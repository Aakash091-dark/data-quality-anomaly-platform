
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.services.ingestion import parse_uploaded_file
from app.services.data_quality import run_data_quality_checks
from app.services.feature_engineering import generate_features
from app.models.train import train_anomaly_model
from app.models.inference import score_anomaly
import mlflow

def load_and_chunk_log(file_path, chunk_size=1000):
    """
    Reads a log file and yields chunks as DataFrames.
    """
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Use the ingestion service logic to get full dataframe
    # The ingestion service expects filename and bytes
    file_name = os.path.basename(file_path)
    # Mocking uploaded file behavior
    if file_name.endswith('.log'):
        # Ingestion service supports .txt, let's treat .log as .txt or extend it
        # The ingestion service in the file read earlier checks extension.
        # It supports txt. Let's assume we pass it as .txt or handle .log manually here.
        # ingestion.py: elif ext == "txt": ...
        # If I pass "foo.log", ingestion might fail if it strictly checks extension.
        # Let's peek ingestion again or just bypass it and create DF directly since we are testing "the whole thing" 
        # but ingestion is part of "the whole thing".
        # Let's try to use ingestion.
        pass

    # Actually, let's just use the logic from ingestion but applied to local file
    # Or modify the filename to end in .txt for the function call
    
    # Better: just read lines directly to control chunking efficiently
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Loaded {file_path}: {total_lines} lines.")
    
    for i in range(0, total_lines, chunk_size):
        chunk_lines = lines[i:i + chunk_size]
        # Create DataFrame as expected by pipeline
        df = pd.DataFrame({"text": [line.strip() for line in chunk_lines]})
        yield df

def run_openstack_test():
    base_dir = Path(__file__).resolve().parent / "OpenStack_log"
    normal_files = [base_dir / "openstack_normal1.log", base_dir / "openstack_normal2.log"]
    abnormal_file = base_dir / "openstack_abnormal.log"

    print("--- Starting OpenStack Log Pipeline Test ---")

    # Set MLflow tracking URI to local directory
    mlflow_dir = Path(__file__).resolve().parent.parent / "mlruns"
    mlflow.set_tracking_uri(f"file:///{str(mlflow_dir).replace(os.sep, '/')}")
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

    # Set experiment to avoid "Could not find experiment with ID 0" error
    try:
        mlflow.set_experiment("openstack_test")
    except Exception as e:
        print(f"Warning: Could not set experiment: {e}")

    # 1. Prepare Training Data
    training_features = []
    
    print("\n[Phase 1] Processing Training Data (Normal Logs)...")
    for log_file in normal_files:
        if not log_file.exists():
            print(f"Warning: {log_file} not found.")
            continue
            
        print(f"Processing {log_file.name}...")
        for chunk_df in load_and_chunk_log(log_file, chunk_size=2000):
            # Pipeline Step 1: Quality Checks
            quality_result = run_data_quality_checks(chunk_df)
            
            # Pipeline Step 2: Feature Engineering
            features = generate_features(chunk_df, quality_result["data_type"])
            training_features.append(features)

    if not training_features:
        print("No training data found. Aborting.")
        return

    print(f"Collected {len(training_features)} training samples.")

    # 2. Train Model
    print("\n[Phase 2] Training Anomaly Model...")
    model = train_anomaly_model(training_features)
    print("Model trained successfully.")

    # 3. Test on Abnormal Data
    print("\n[Phase 3] Processing Test Data (Abnormal Log)...")
    if not abnormal_file.exists():
        print(f"Error: {abnormal_file} not found.")
        return

    anomaly_counts = {"normal": 0, "anomaly": 0}
    severity_counts = {"low": 0, "medium": 0, "high": 0}

    print(f"Processing {abnormal_file.name}...")
    for i, chunk_df in enumerate(load_and_chunk_log(abnormal_file, chunk_size=2000)):
        # Pipeline Step 1: Quality Checks
        quality_result = run_data_quality_checks(chunk_df)
        
        # Pipeline Step 2: Feature Engineering
        features = generate_features(chunk_df, quality_result["data_type"])
        
        # Pipeline Step 3: Inference
        result = score_anomaly(model, features)
        
        pred = result["prediction"]
        sev = result["severity"]
        
        anomaly_counts[pred] = anomaly_counts.get(pred, 0) + 1
        if pred == "anomaly":
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            # print(f"Chunk {i}: Anomaly Detected ({sev})")

    print("\n[Phase 4] Results Summary")
    print(f"Total Test Chunks Processed: {sum(anomaly_counts.values())}")
    print(f"Normal Predictions: {anomaly_counts.get('normal', 0)}")
    print(f"Anomaly Predictions: {anomaly_counts.get('anomaly', 0)}")
    print(f"Severity Distribution: {severity_counts}")

    if anomaly_counts.get('anomaly', 0) > 0:
        print("\nSUCCESS: Anomalies detected in abnormal log.")
    else:
        print("\nWARNING: No anomalies detected. Model might be insensitive or features insufficient.")

if __name__ == "__main__":
    run_openstack_test()
