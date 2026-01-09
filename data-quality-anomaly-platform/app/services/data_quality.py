import pandas as pd
from typing import Dict, Any


# -----------------------------
# Structured Data Checks
# -----------------------------

def check_structured_data(df: pd.DataFrame) -> Dict[str, Any]:
    report = {}

    report["row_count"] = df.shape[0]
    report["column_count"] = df.shape[1]

    report["missing_values"] = df.isnull().sum().to_dict()
    report["duplicate_rows"] = int(df.duplicated().sum())

    report["data_types"] = df.dtypes.astype(str).to_dict()

    report["empty_columns"] = [
        col for col in df.columns if df[col].isnull().all()
    ]

    return report


# -----------------------------
# Unstructured / Text Data Checks
# -----------------------------

def check_text_data(df: pd.DataFrame) -> Dict[str, Any]:
    text_col = df.columns[0]

    lengths = df[text_col].dropna().apply(len)

    report = {
        "total_lines": len(df),
        "empty_lines": int(df[text_col].isnull().sum()),
        "avg_text_length": float(lengths.mean()) if not lengths.empty else 0,
        "min_text_length": int(lengths.min()) if not lengths.empty else 0,
        "max_text_length": int(lengths.max()) if not lengths.empty else 0,
    }

    return report


# -----------------------------
# Dispatcher (Auto-detect type)
# -----------------------------

def run_data_quality_checks(df: pd.DataFrame) -> Dict[str, Any]:
    if df.shape[1] == 1 and df.columns[0] == "text":
        return {
            "data_type": "unstructured",
            "quality_report": check_text_data(df)
        }

    return {
        "data_type": "structured",
        "quality_report": check_structured_data(df)
    }
