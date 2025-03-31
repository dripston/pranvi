from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uvicorn

from weather_predictor import WeatherPredictor

app = FastAPI(title="Weather Prediction AI Agent")

class WeatherRequest(BaseModel):
    location: str
    date: str

class WeatherResponse(BaseModel):
    location: str
    date: str
    forecast: dict

weather_predictor = WeatherPredictor()

@app.post("/predict_weather", response_model=WeatherResponse)
async def predict_weather(request: WeatherRequest):
    try:
        forecast = weather_predictor.predict(request.location, request.date)
        return WeatherResponse(
            location=request.location,
            date=request.date,
            forecast=forecast
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 