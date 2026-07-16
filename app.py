"""
Stock Price Prediction Dashboard — FastAPI Backend
Run: python app.py
"""

import os
import io
import csv
import json
import logging
import uvicorn
import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

import ml_engine as ml

log = logging.getLogger(__name__)

# ── App Init ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Stock AI Dashboard", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state (lightweight — fine for college project scope)
STATE: dict = {"df": None, "models": None, "scaler": None, "cache": None,
               "metrics": {}, "dataset_path": "dataset/stock_data.csv",
               "best_model": "LinearRegression"}

# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """On start: load data → load/train models."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    path = STATE["dataset_path"]
    if not os.path.exists(path):
        _generate_sample_dataset(path)
        log.info("Sample dataset generated.")

    try:
        STATE["df"] = ml.load_and_clean(path)
    except Exception as e:
        log.error(f"Dataset load failed: {e}")
        return

    if ml.models_exist():
        STATE["models"], STATE["scaler"], STATE["cache"] = ml.load_models()
        STATE["metrics"] = STATE["cache"].get("results", {})
    else:
        log.info("Training models — this may take a minute…")
        STATE["metrics"] = ml.train_all(STATE["df"])
        STATE["models"], STATE["scaler"], STATE["cache"] = ml.load_models()

    # Pick best model by lowest RMSE
    best = min(
        ((n, v["metrics"].get("RMSE", 9e9)) for n, v in STATE["metrics"].items() if "metrics" in v),
        key=lambda x: x[1], default=("LinearRegression", 0)
    )
    STATE["best_model"] = best[0]
    log.info(f"Best model: {STATE['best_model']}")


def _generate_sample_dataset(path: str):
    """Generate synthetic AAPL-like stock data for demo."""
    import random, math
    random.seed(42)
    rows = []
    price = 150.0
    for i in range(1000):
        date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=i)
        if date.weekday() >= 5:
            continue
        change = random.gauss(0.0005, 0.018)
        price = max(10, price * (1 + change))
        h = price * (1 + abs(random.gauss(0, 0.008)))
        l = price * (1 - abs(random.gauss(0, 0.008)))
        o = l + (h - l) * random.random()
        vol = int(random.gauss(80_000_000, 20_000_000))
        rows.append([date.strftime("%Y-%m-%d"), round(o,2), round(h,2), round(l,2),
                     round(price,2), round(price,2), max(0, vol)])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows, columns=["Date","Open","High","Low","Close","Adj Close","Volume"])
    df.to_csv(path, index=False)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_df():
    if STATE["df"] is None:
        raise HTTPException(400, "No dataset loaded.")
    return STATE["df"]

def _require_models():
    if STATE["models"] is None:
        raise HTTPException(400, "Models not trained yet.")
    return STATE["models"]

# ── API Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/api/summary")
async def summary():
    df = _require_df()
    ins = ml.generate_insights(df)
    best = STATE["best_model"]
    best_rmse = STATE["metrics"].get(best, {}).get("metrics", {}).get("RMSE", 0)
    return {**ins, "best_model": best, "best_rmse": best_rmse,
            "models_trained": len(STATE["metrics"])}


@app.get("/api/analytics")
async def analytics():
    df = _require_df()
    tail = df.tail(200)
    return {
        "dates": tail["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "close": tail["Close"].round(4).tolist(),
        "open": tail["Open"].round(4).tolist(),
        "high": tail["High"].round(4).tolist(),
        "low": tail["Low"].round(4).tolist(),
        "volume": tail["Volume"].tolist(),
        "ma7": tail["MA7"].round(4).tolist(),
        "ma30": tail["MA30"].round(4).tolist(),
        "daily_return": tail["DailyReturn"].round(6).tolist(),
        "volatility": tail["Volatility"].round(6).tolist(),
        "momentum": tail["Momentum"].round(4).tolist(),
    }


@app.get("/api/metrics")
async def metrics():
    return {"metrics": STATE["metrics"], "best_model": STATE["best_model"]}


@app.get("/api/dataset")
async def dataset_preview(page: int = 1, per_page: int = 20, search: str = ""):
    df = _require_df()
    if search:
        df = df[df["Date"].astype(str).str.contains(search, case=False)]
    total = len(df)
    start = (page - 1) * per_page
    slice_df = df.iloc[start:start + per_page]
    cols = ["Date","Open","High","Low","Close","Volume","MA7","MA30","DailyReturn","Volatility"]
    slice_df = slice_df[[c for c in cols if c in slice_df.columns]]
    slice_df["Date"] = slice_df["Date"].dt.strftime("%Y-%m-%d")
    return {"data": slice_df.round(4).to_dict(orient="records"),
            "total": total, "page": page, "per_page": per_page,
            "pages": (total + per_page - 1) // per_page}


@app.get("/api/predictions_chart")
async def predictions_chart():
    best = STATE["best_model"]
    cache = STATE.get("cache", {})
    results = cache.get("results", STATE["metrics"])
    model_data = results.get(best, {})
    preds = model_data.get("predictions", [])
    actuals = model_data.get("actuals", [])
    n = min(len(preds), len(actuals), 100)
    return {"predictions": [round(p, 4) for p in preds[-n:]],
            "actuals": [round(a, 4) for a in actuals[-n:]],
            "model": best}


class PredictRequest(BaseModel):
    Open: float
    High: float
    Low: float
    Volume: float
    MA7: float
    MA30: float
    Momentum: float
    Volatility: float
    DailyReturn: float
    Close: Optional[float] = None
    model_name: Optional[str] = None


@app.post("/api/predict")
async def predict(req: PredictRequest):
    models = _require_models()
    model_name = req.model_name or STATE["best_model"]
    if model_name not in models:
        raise HTTPException(400, f"Unknown model: {model_name}")
    features = req.dict()
    result = ml.predict_next(features, models[model_name], STATE["scaler"], model_name, STATE["cache"])
    return result


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted.")
    content = await file.read()
    path = STATE["dataset_path"]
    with open(path, "wb") as f:
        f.write(content)
    try:
        STATE["df"] = ml.load_and_clean(path)
    except Exception as e:
        raise HTTPException(400, f"Invalid dataset: {e}")
    # Force retrain
    for fname in ["model_linear.pkl","model_rf.pkl","model_xgb.pkl","model_lstm.keras","scaler.pkl","eval_cache.pkl"]:
        fpath = os.path.join("models", fname)
        if os.path.exists(fpath):
            os.remove(fpath)
    STATE["models"] = None
    return {"message": f"Dataset uploaded: {len(STATE['df'])} rows. Trigger /api/retrain to retrain."}


@app.post("/api/retrain")
async def retrain():
    df = _require_df()
    STATE["metrics"] = ml.train_all(df)
    STATE["models"], STATE["scaler"], STATE["cache"] = ml.load_models()
    best = min(
        ((n, v["metrics"].get("RMSE", 9e9)) for n, v in STATE["metrics"].items() if "metrics" in v),
        key=lambda x: x[1], default=("LinearRegression", 0)
    )
    STATE["best_model"] = best[0]
    return {"message": "Retraining complete", "best_model": STATE["best_model"]}


@app.get("/api/download/dataset")
async def download_dataset():
    df = _require_df()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(buf, media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=clean_dataset.csv"})


@app.get("/api/download/report")
async def download_report():
    df = _require_df()
    ins = ml.generate_insights(df)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Metric", "Value"])
    for k, v in ins.items():
        w.writerow([k, v])
    w.writerow([])
    w.writerow(["Model", "RMSE", "MAE", "R2", "TrainTime"])
    for name, data in STATE["metrics"].items():
        m = data.get("metrics", {})
        w.writerow([name, m.get("RMSE",""), m.get("MAE",""), m.get("R2",""), m.get("TrainTime","")])
    buf.seek(0)
    return StreamingResponse(buf, media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=report.csv"})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
