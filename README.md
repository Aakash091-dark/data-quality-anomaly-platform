# 🚀 AI-Based Data Quality & Anomaly Detection Platform

An **enterprise-grade AI platform** that ingests data from multiple formats, performs automated data quality validation, detects anomalies using machine learning, explains root causes with Explainable AI, and exposes everything via a production-ready FastAPI service.

This project is designed to reflect **real-world data observability systems** used in large-scale production environments.

---

## 📌 Problem Statement

In real organizations:
- Bad data silently enters pipelines
- Dashboards and ML models consume corrupted data
- Issues are detected late
- Root-cause analysis is manual and slow

This platform solves that by providing:
- Automated data quality checks
- ML-based anomaly detection
- Explainable AI for trust and debugging
- Production-ready APIs and MLOps workflows

---

## 🎯 Key Features

✅ Universal data ingestion (CSV, JSON, Excel, TXT, PDF)  
✅ Automated rule-based data quality checks  
✅ ML-based anomaly detection (Isolation Forest)  
✅ Explainable anomaly detection using SHAP  
✅ Supports structured & unstructured data  
✅ FastAPI-based production service  
✅ ML lifecycle management with MLflow  
✅ Dockerized deployment  
✅ End-to-end pipeline testing  

---


---

## 🧠 Tech Stack

| Layer | Technology |
|------|-----------|
| Backend API | FastAPI |
| Data Processing | Pandas, NumPy |
| ML | Scikit-learn (Isolation Forest) |
| Explainability | SHAP |
| MLOps | MLflow |
| Containerization | Docker, Docker Compose |
| Testing | PyTest |
| Data Formats | CSV, JSON, Excel, TXT, PDF |

---


---

## 🔄 Pipeline Explanation (Step-by-Step)

### 1️⃣ Universal Ingestion
- Accepts CSV, JSON, Excel, TXT, PDF
- Automatically detects file type
- Converts input into a unified internal format

### 2️⃣ Data Quality Engine
- Missing value detection
- Duplicate row detection
- Schema & data type checks
- Text-specific quality metrics for unstructured data

### 3️⃣ Feature Engineering
- Aggregated statistical features for structured data
- Text-length and density features for unstructured data
- Produces stable ML-ready feature vectors

### 4️⃣ ML-Based Anomaly Detection
- Isolation Forest trained on historical feature distributions
- Detects unknown and unexpected data behavior
- Outputs anomaly score and severity

### 5️⃣ Explainable AI
- SHAP-based feature contribution analysis
- Explains why a data batch was flagged as anomalous
- Enables root-cause analysis

### 6️⃣ MLOps with MLflow
- Experiment tracking
- Model versioning
- Model registry with Production stage
- Safe model loading in APIs

---

## 🔌 API Usage

### Upload Data
```http
POST /api/upload
Content-Type: multipart/form-data




