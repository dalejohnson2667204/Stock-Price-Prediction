"""
ML Engine: Data processing, feature engineering, model training & evaluation.
Handles all ML operations for the Stock Prediction Dashboard.
"""

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Optional

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Paths
MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_COLS = ["Open", "High", "Low", "Volume", "MA7", "MA30", "Momentum", "Volatility", "DailyReturn"]
TARGET_COL = "Close"

# ── Data ──────────────────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, clean, engineer features, return ready DataFrame."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names
    rename = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "").replace("_", "")
        if cl == "adjclose":
            rename[c] = "AdjClose"
        elif cl in ("open", "high", "low", "close", "volume", "date"):
            rename[c] = cl.capitalize()
    df.rename(columns=rename, inplace=True)

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.drop_duplicates(subset=["Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Feature engineering
    df["MA7"] = df["Close"].rolling(7).mean()
    df["MA30"] = df["Close"].rolling(30).mean()
    df["DailyReturn"] = df["Close"].pct_change()
    df["Volatility"] = df["DailyReturn"].rolling(7).std()
    df["Momentum"] = df["Close"] - df["Close"].shift(10)
    df["PriceChangePct"] = df["Close"].pct_change() * 100
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info(f"Dataset ready: {len(df)} rows")
    return df


def split_and_scale(df: pd.DataFrame):
    """80/20 split + MinMax scaling. Returns X_train, X_test, y_train, y_test, scaler."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    return X_train, X_test, y_train, y_test, scaler

# ── Models ─────────────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred, t_train=0.0, t_pred=0.0) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4), "R2": round(r2, 4),
            "TrainTime": round(t_train, 3), "PredTime": round(t_pred, 5)}


def train_linear(X_train, y_train, X_test, y_test):
    path = os.path.join(MODEL_DIR, "model_linear.pkl")
    t0 = time.time(); m = LinearRegression().fit(X_train, y_train); t_train = time.time() - t0
    t0 = time.time(); preds = m.predict(X_test); t_pred = time.time() - t0
    joblib.dump(m, path)
    log.info("Linear Regression trained")
    return m, preds, _metrics(y_test, preds, t_train, t_pred)


def train_rf(X_train, y_train, X_test, y_test):
    path = os.path.join(MODEL_DIR, "model_rf.pkl")
    t0 = time.time(); m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train); t_train = time.time() - t0
    t0 = time.time(); preds = m.predict(X_test); t_pred = time.time() - t0
    joblib.dump(m, path)
    log.info("Random Forest trained")
    return m, preds, _metrics(y_test, preds, t_train, t_pred)


def train_xgb(X_train, y_train, X_test, y_test):
    path = os.path.join(MODEL_DIR, "model_xgb.pkl")
    t0 = time.time(); m = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42).fit(X_train, y_train); t_train = time.time() - t0
    t0 = time.time(); preds = m.predict(X_test); t_pred = time.time() - t0
    joblib.dump(m, path)
    log.info("XGBoost trained")
    return m, preds, _metrics(y_test, preds, t_train, t_pred)


def train_lstm(X_train, y_train, X_test, y_test):
    """Lightweight LSTM (TF/Keras)."""
    import tensorflow as tf
    path = os.path.join(MODEL_DIR, "model_lstm.keras")

    X_tr = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_te = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    m = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    m.compile(optimizer="adam", loss="mse")

    t0 = time.time()
    m.fit(X_tr, y_train, epochs=20, batch_size=32, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    t_train = time.time() - t0

    t0 = time.time(); preds = m.predict(X_te, verbose=0).flatten(); t_pred = time.time() - t0
    m.save(path)
    log.info("LSTM trained")
    return m, preds, _metrics(y_test, preds, t_train, t_pred)

# ── Orchestration ──────────────────────────────────────────────────────────────

def models_exist() -> bool:
    files = ["model_linear.pkl", "model_rf.pkl", "model_xgb.pkl", "model_lstm.keras", "scaler.pkl"]
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in files)


def train_all(df: pd.DataFrame) -> dict:
    """Train all models, return results dict."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    results = {}
    for name, fn in [("LinearRegression", train_linear), ("RandomForest", train_rf),
                     ("XGBoost", train_xgb), ("LSTM", train_lstm)]:
        try:
            _, preds, metrics = fn(X_train, y_train, X_test, y_test)
            results[name] = {"metrics": metrics, "predictions": preds.tolist(),
                             "actuals": y_test.tolist()}
        except Exception as e:
            log.error(f"{name} failed: {e}")
            results[name] = {"metrics": {}, "error": str(e)}

    # Save test actuals for chart use
    joblib.dump({"y_test": y_test, "results": results}, os.path.join(MODEL_DIR, "eval_cache.pkl"))
    log.info("All models trained and saved.")
    return results


def load_models() -> tuple:
    """Load all saved models and scaler."""
    import tensorflow as tf
    lin = joblib.load(os.path.join(MODEL_DIR, "model_linear.pkl"))
    rf  = joblib.load(os.path.join(MODEL_DIR, "model_rf.pkl"))
    xgb = joblib.load(os.path.join(MODEL_DIR, "model_xgb.pkl"))
    lstm= tf.keras.models.load_model(os.path.join(MODEL_DIR, "model_lstm.keras"))
    scaler = joblib.load(SCALER_PATH)
    cache = joblib.load(os.path.join(MODEL_DIR, "eval_cache.pkl"))
    log.info("Models loaded from disk.")
    return {"LinearRegression": lin, "RandomForest": rf, "XGBoost": xgb, "LSTM": lstm}, scaler, cache


def predict_next(features: dict, model, scaler, model_name: str, eval_cache: dict) -> dict:
    """Predict next-day close price."""
    row = np.array([[features[c] for c in FEATURE_COLS]])
    row_scaled = scaler.transform(row)

    if model_name == "LSTM":
        import tensorflow as tf
        row_scaled = row_scaled.reshape(1, 1, row_scaled.shape[1])
        pred = float(model.predict(row_scaled, verbose=0)[0][0])
    else:
        pred = float(model.predict(row_scaled)[0])

    # Confidence: 1 - normalised RMSE from cached metrics
    results = eval_cache.get("results", {})
    rmse = results.get(model_name, {}).get("metrics", {}).get("RMSE", 0)
    actuals = eval_cache.get("y_test", [])
    price_range = float(np.max(actuals) - np.min(actuals)) if len(actuals) else 1
    confidence = max(0, min(100, round((1 - rmse / price_range) * 100, 1))) if price_range else 70.0

    curr_close = features.get("Close", pred)
    pct_change = round((pred - curr_close) / (curr_close + 1e-8) * 100, 2)
    trend = "Bullish 📈" if pct_change > 0.5 else "Bearish 📉" if pct_change < -0.5 else "Sideways ↔"

    return {"predicted_price": round(pred, 4), "trend": trend,
            "confidence": confidence, "pct_change": pct_change, "model_used": model_name}

# ── Insights ───────────────────────────────────────────────────────────────────

def generate_insights(df: pd.DataFrame) -> dict:
    """Generate market insights from the DataFrame."""
    close = df["Close"]
    ma7, ma30 = df["MA7"].iloc[-1], df["MA30"].iloc[-1]
    last = close.iloc[-1]
    ret = df["DailyReturn"].dropna()
    vol = df["Volatility"].dropna().iloc[-1]
    mom = df["Momentum"].iloc[-1]

    # Trend
    if ma7 > ma30 and last > ma7:
        trend = "Bullish"
    elif ma7 < ma30 and last < ma7:
        trend = "Bearish"
    else:
        trend = "Sideways"

    # Recommendation
    score = 0
    if ma7 > ma30: score += 1
    if last > ma7: score += 1
    if mom > 0: score += 1
    if ret.mean() > 0: score += 1
    recs = {4: "Strong Buy 🚀", 3: "Buy 📈", 2: "Hold ⚖️", 1: "Sell 📉", 0: "Strong Sell 🔴"}
    recommendation = recs.get(score, "Hold ⚖️")

    return {
        "trend": trend,
        "recommendation": recommendation,
        "highest_close": round(float(close.max()), 4),
        "lowest_close": round(float(close.min()), 4),
        "avg_close": round(float(close.mean()), 4),
        "last_close": round(float(last), 4),
        "ma7": round(float(ma7), 4),
        "ma30": round(float(ma30), 4),
        "avg_daily_return": round(float(ret.mean() * 100), 4),
        "volatility": round(float(vol * 100), 4),
        "momentum": round(float(mom), 4),
        "total_rows": len(df),
        "score": score,
    }
