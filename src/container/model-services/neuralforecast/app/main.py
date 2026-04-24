import os
from typing import List, Dict, Optional, Any
from threading import Thread, Lock
from uuid import uuid4
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

from .model import NeuralForecastManager

training_02 = False


# Directory for model persistence (mounted via volume)
MODELS_DIR = os.getenv("MODELS_DIR", "/models")
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI(title="NeuralForecast Service")

# Initialize manager
manager = NeuralForecastManager(models_dir=MODELS_DIR)

# Simple in-memory job registry for training tasks
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = Lock()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _run_training_job(job_id: str, fit_args: Dict[str, Any]) -> None:
    # Update status to running
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        job["started_at"] = _now_iso()
    try:
        result = manager.fit_and_save(**fit_args)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["status"] = "completed"
                job["finished_at"] = _now_iso()
                job["result"] = result
    except Exception as e:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["finished_at"] = _now_iso()
                job["error"] = str(e)


# API schemas
class TimeSeriesItem(BaseModel):
    unique_id: str
    ds: str
    y: Optional[float] = None
    exog: Dict[str, float] = {}


class TrainRequest(BaseModel):
    data: List[TimeSeriesItem]
    freq: str
    h: int
    input_size: int
    futr_exog: List[str] = []
    hist_exog: List[str] = []
    stat_exog: List[str] = []
    use_auto: bool = False
    num_samples: int = 5


class CrossValidationRequest(BaseModel):
    train_data: List[TimeSeriesItem]
    test_data: List[TimeSeriesItem]
    freq: str
    h: int
    input_size: int
    futr_exog: List[str] = []
    hist_exog: List[str] = []
    stat_exog: List[str] = []
    use_auto: bool = False
    num_samples: int = 5
    val_size: Optional[int] = None
    test_size: Optional[int] = None
    step_size: int = 1
    n_windows: Optional[int] = None


class ForecastRequest(BaseModel):
    data: List[TimeSeriesItem]
    static: Optional[List[Dict[str, float]]] = None


@app.post("/train")
def train(model_name: str = Query(..., min_length=1), req: TrainRequest = ...):
    # Build dataframes synchronously to validate input early
    df = manager.to_df(req.data, include_y=True)
    static_df = manager.static_df_from(df, req.stat_exog)

    # Prepare args for background training
    fit_args: Dict[str, Any] = dict(
        df=df,
        static_df=static_df,
        freq=req.freq,
        model_name=model_name,
        h=req.h,
        input_size=req.input_size,
        futr_exog=req.futr_exog,
        hist_exog=req.hist_exog,
        stat_exog=req.stat_exog,
        use_auto=req.use_auto,
        num_samples=req.num_samples,
    )

    job_id = str(uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "model_name": model_name,
            "status": "queued",
            "created_at": _now_iso(),
        }

    t = Thread(target=_run_training_job, args=(job_id, fit_args), daemon=True)
    t.start()

    return {"job_id": job_id, "status": "queued", "model_name": model_name}


@app.post("/cross_validation")
def cross_validation(req: CrossValidationRequest = ...):
    # Build dataframes synchronously to validate input early
    df_train = manager.to_df(req.train_data, include_y=True)
    df_test = manager.to_df(req.test_data, include_y=True)
    df = pd.concat([df_train, df_test])
    static_df = manager.static_df_from(df, req.stat_exog)

    if training_02 is False:
        # Calculate val_size and test_size if not provided
        val_size = req.val_size if req.val_size is not None else len(req.test_data) // 2
        test_size = req.test_size if req.test_size is not None else len(req.test_data)
    else:   
        # Calculate val_size and test_size if not provided
        val_size = req.val_size if req.val_size is not None else int(len(req.train_data) * 0.1)
        test_size = req.test_size if req.test_size is not None else len(req.test_data)



    # Prepare args for CV
    cv_args: Dict[str, Any] = dict(
        df=df,
        static_df=static_df,
        freq=req.freq,
        h=req.h,
        input_size=req.input_size,
        futr_exog=req.futr_exog,
        hist_exog=req.hist_exog,
        stat_exog=req.stat_exog,
        use_auto=req.use_auto,
        num_samples=req.num_samples,
        val_size=val_size,
        test_size=test_size,
        step_size=req.step_size,
        n_windows=req.n_windows,
    )

    result = manager.cross_validate(**cv_args)
    return result.to_dict(orient="records")


@app.post("/forecast")
def forecast(model_name: str = Query(..., min_length=1), req: ForecastRequest = ...):
    df = manager.to_df(req.data, include_y=True)
    # futr_df = manager.to_df([], include_y=False)
    static_df = manager.static_df_from_rows(req.static)
    preds = manager.predict(model_name=model_name, df=df, futr_df=None, static_df=static_df)
    return preds.to_dict(orient="records")


@app.get("/train/status")
def train_status(job_id: str = Query(..., min_length=1)):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return job


@app.get("/models")
def list_models():
    return {"models": manager.list_models()} 


@app.get("/health")
def health():
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        return {"status": "ok", "models_dir": MODELS_DIR}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))