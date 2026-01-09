from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# -----------------------------
# JSON / API-based ingestion
# -----------------------------

class JSONIngestRequest(BaseModel):
    """
    Used when data is sent directly as JSON
    """
    records: List[Dict[str, Any]] = Field(
        ...,
        description="List of records (each record is a key-value map)"
    )


class JSONIngestResponse(BaseModel):
    """
    Metadata returned after JSON ingestion
    """
    status: str
    rows_received: int
    columns: List[str]


# -----------------------------
# File upload ingestion
# -----------------------------

class FileIngestResponse(BaseModel):
    """
    Metadata returned after file upload ingestion
    """
    status: str
    file_name: str
    file_type: str
    rows: Optional[int] = None
    columns: Optional[List[str]] = None
    message: Optional[str] = None


# -----------------------------
# Unified internal schema
# (used later in pipeline)
# -----------------------------

class StandardizedDataSchema(BaseModel):
    """
    Internal representation after ingestion
    """
    data_type: str = Field(
        ...,
        description="structured | unstructured"
    )
    row_count: Optional[int]
    column_count: Optional[int]
    preview: Optional[List[Dict[str, Any]]]


# -----------------------------
# Error schema (clean API errors)
# -----------------------------

class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str
