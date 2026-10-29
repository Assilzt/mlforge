from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from mlforge_core import profile_dataframe, train_dataframe

app = FastAPI(title="MLForge API", version="0.1.0")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "mlforge"}


@app.post("/api/profile")
async def profile(file: UploadFile = File(...)):  # noqa: B008
    if Path(file.filename or "").suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, "Only CSV and Excel files are supported")
    path = DATA_DIR / f"{uuid.uuid4().hex}_{Path(file.filename or 'upload.csv').name}"
    with path.open("wb") as handle: shutil.copyfileobj(file.file, handle)
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    return {"file": str(path), "profile": profile_dataframe(df)}


@app.post("/api/train")
async def train(file: UploadFile = File(...), target: str = ""):  # noqa: B008
    if not target: raise HTTPException(400, "target is required")
    path = DATA_DIR / f"{uuid.uuid4().hex}_{Path(file.filename or 'upload.csv').name}"
    with path.open("wb") as handle: shutil.copyfileobj(file.file, handle)
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    out = DATA_DIR / uuid.uuid4().hex
    try: report = train_dataframe(df, target, out)
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc
    return {"result_dir": str(out), "report": report}


@app.get("/api/results/{result_id}/{filename}")
def result_file(result_id: str, filename: str):
    if Path(filename).name != filename or filename not in {"model.joblib", "report.json", "predictions.csv"}: raise HTTPException(400, "Invalid file")
    path = DATA_DIR / result_id / filename
    if not path.exists(): raise HTTPException(404, "Result not found")
    return FileResponse(path)
