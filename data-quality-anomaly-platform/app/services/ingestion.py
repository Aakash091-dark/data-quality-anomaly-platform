import pandas as pd
from io import BytesIO, StringIO
import json
from PyPDF2 import PdfReader

def parse_uploaded_file(file_name: str, content: bytes):
    ext = file_name.split(".")[-1].lower()

    if ext == "csv":
        return pd.read_csv(StringIO(content.decode()))

    elif ext in ["xls", "xlsx"]:
        return pd.read_excel(BytesIO(content))

    elif ext == "json":
        data = json.loads(content.decode())
        return pd.DataFrame(data)

    elif ext == "txt":
        text = content.decode()
        return pd.DataFrame({"text": text.splitlines()})

    elif ext == "pdf":
        reader = PdfReader(BytesIO(content))
        text = " ".join(page.extract_text() for page in reader.pages)
        return pd.DataFrame({"text": text.split("\n")})

    else:
        raise ValueError("Unsupported file format")
