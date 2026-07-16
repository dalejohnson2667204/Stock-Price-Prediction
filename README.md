# Stock Price Prediction Dashboard

An AI-powered dashboard that predicts next-day stock prices. It engineers technical
features from OHLCV data, trains multiple ML models, and serves everything through an
interactive analytics web UI.

## Demo

[Watch the demo video](https://github.com/dalejohnson2667204/Stock-Price-Prediction/blob/main/stock.prediction.mp4)

## Features

- Feature engineering: MA7, MA30, momentum, volatility, daily returns
- Trains & compares 4 models: Linear Regression, Random Forest, XGBoost, LSTM
- Auto-selects the best model by RMSE
- Live charts, model comparison table, and predicted-vs-actual chart
- Prediction form with autofill from latest data row
- Upload your own CSV and retrain on demand
- Export cleaned dataset & report as CSV

## Tech Stack

**Backend:** FastAPI, scikit-learn, XGBoost, TensorFlow/Keras, pandas
**Frontend:** HTML, CSS, JS, Chart.js

## Run Locally

```bash
git clone https://github.com/yourusername/stock-prediction-dashboard.git
cd stock-prediction-dashboard
pip install fastapi "uvicorn[standard]" python-multipart pandas numpy scikit-learn xgboost tensorflow joblib
python app.py
```
Then open **http://localhost:8000**.

## Key API Endpoints

| Endpoint | Description |
|---|---|
| `/api/summary` | Market overview & recommendation |
| `/api/metrics` | Model performance comparison |
| `/api/predict` | Predict next-day close price |
| `/api/upload` | Upload a new CSV dataset |
| `/api/retrain` | Retrain all models |

## Disclaimer

For educational purposes only — not financial advice.
