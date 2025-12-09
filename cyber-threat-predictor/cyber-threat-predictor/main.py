from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from prophet import Prophet

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ThreatDataItem(BaseModel):
    type: str
    count: int
    date: str  # format YYYY-MM-DD

class PredictRequest(BaseModel):
    data: List[ThreatDataItem]
    forecastDays: int

@app.post("/predict")
async def predict_threats(request: PredictRequest):
    try:
        if not request.data or request.forecastDays < 1:
            return {"error": "No data provided or invalid forecastDays."}

        df = pd.DataFrame([item.dict() for item in request.data])
        if df.empty:
            return {"error": "Input data is empty."}

        predictions = []

        for threat_type in df['type'].unique():
            threat_df = df[df['type'] == threat_type][['date', 'count']].rename(columns={'date':'ds','count':'y'})
            
            # Convert dates safely
            try:
                threat_df['ds'] = pd.to_datetime(threat_df['ds'])
            except Exception:
                # If date conversion fails, set dummy dates
                threat_df['ds'] = pd.date_range(start=pd.Timestamp.today(), periods=len(threat_df))

            # If less than 2 data points, use simple mean
            if len(threat_df) < 2:
                forecast_value = int(threat_df['y'].mean() * request.forecastDays)
                predictions.append({"type": threat_type, "forecast": forecast_value})
                continue

            # Fit Prophet model
            try:
                model = Prophet(daily_seasonality=True)
                model.fit(threat_df)
                future = model.make_future_dataframe(periods=request.forecastDays)
                forecast = model.predict(future)
                forecast_sum = int(forecast['yhat'].tail(request.forecastDays).sum())
                predictions.append({"type": threat_type, "forecast": forecast_sum})
            except Exception:
                # If Prophet fails, fallback to simple mean
                forecast_value = int(threat_df['y'].mean() * request.forecastDays)
                predictions.append({"type": threat_type, "forecast": forecast_value})

        predictions.sort(key=lambda x: x["forecast"], reverse=True)
        return predictions

    except Exception as e:
        # Catch-all to prevent backend crash
        return {"error": str(e)}
