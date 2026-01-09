import os
from pathlib import Path

# Define the base directory
base_dir = Path("data-quality-anomaly-platform")

# Folder structure definition
structure = {
    "app": {
        "main.py": None,
        "api": {
            "__init__.py": None,
            "routes.py": None,
            "schemas.py": None,
        },
        "core": {
            "__init__.py": None,
            "config.py": None,
            "logger.py": None,
        },
        "services": {
            "__init__.py": None,
            "data_quality.py": None,
            "feature_engineering.py": None,
            "anomaly_service.py": None,
            "explainability.py": None,
        },
        "models": {
            "__init__.py": None,
            "train.py": None,
            "inference.py": None,
            "registry.py": None,
        },
    },
    "ml": {
        "anomaly": {
            "__init__.py": None,
            "isolation_forest.py": None,
            "autoencoder.py": None,
            "ensemble.py": None,
        },
        "explain": {
            "__init__.py": None,
            "shap_explainer.py": None,
        },
    },
    "data": {
        "raw": {},
        "processed": {},
        "baseline": {},
    },
    "mlflow": {
        "experiments": {},
    },
    "tests": {
        "__init__.py": None,
        "test_pipeline.py": None,
    },
    "docker": {
        "Dockerfile": None,
        "docker-compose.yml": None,
    },
    "scripts": {
        "load_sample_data.py": None,
        "train_model.py": None,
    },
    ".env": None,
    ".gitignore": None,
    "requirements.txt": None,
    "README.md": None,
    "run.sh": None,
}

def create_structure(base, struct):
    for name, content in struct.items():
        path = base / name
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_structure(path, content)
        else:
            path.touch(exist_ok=True)

# Create the folder structure
create_structure(base_dir, structure)

print(f"Project structure created under: {base_dir.resolve()}")