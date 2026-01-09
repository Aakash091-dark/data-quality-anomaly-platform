import pandas as pd
import numpy as np
from typing import Dict, Any


# -----------------------------
# Structured Data Features
# -----------------------------

def structured_features(df: pd.DataFrame) -> Dict[str, Any]:
    features = {}

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        features[f"{col}_mean"] = float(df[col].mean())
        features[f"{col}_std"] = float(df[col].std())
        features[f"{col}_min"] = float(df[col].min())
        features[f"{col}_max"] = float(df[col].max())
        features[f"{col}_missing_ratio"] = float(df[col].isnull().mean())

    features["row_count"] = df.shape[0]
    features["column_count"] = df.shape[1]

    return features


# -----------------------------
# Text / Unstructured Features
# -----------------------------

def text_features(df: pd.DataFrame) -> Dict[str, Any]:
    text_col = df.columns[0]
    texts = df[text_col].dropna()

    lengths = texts.apply(len)

    features = {
        "line_count": len(df),
        "avg_text_length": float(lengths.mean()) if not lengths.empty else 0,
        "std_text_length": float(lengths.std()) if not lengths.empty else 0,
        "min_text_length": int(lengths.min()) if not lengths.empty else 0,
        "max_text_length": int(lengths.max()) if not lengths.empty else 0,
        "empty_line_ratio": float(df[text_col].isnull().mean()),
    }

    return features


# -----------------------------
# Dispatcher
# -----------------------------

def generate_features(df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
    if data_type == "unstructured":
        return text_features(df)

    return structured_features(df)
