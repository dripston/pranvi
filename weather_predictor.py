import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from data_fetcher import OpenMeteoFetcher

class WeatherPredictor:
    def __init__(self):
        self.data_fetcher = OpenMeteoFetcher()
        self.models = {
            'temperature': GradientBoostingRegressor(),
            'precipitation': GradientBoostingRegressor(),
            'humidity': GradientBoostingRegressor(),
            'wind_speed': GradientBoostingRegressor()
        }
        self.scalers = {
            'temperature': StandardScaler(),
            'precipitation': StandardScaler(),
            'humidity': StandardScaler(),
            'wind_speed': StandardScaler()
        }
        # Initialize with Mumbai's coordinates
        self.train(19.0760, 72.8777)

    def train(self, location_lat: float, location_lon: float):
        # Fetch last 30 days of data for training
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        historical_data = self.data_fetcher.fetch_historical_data(
            location_lat, location_lon, start_date, end_date
        )
        
        # Clean data
        historical_data = self._clean_data(historical_data)
        
        # Prepare features
        features = self._prepare_features(historical_data)
        
        # Train models for each weather parameter
        for param in self.models.keys():
            X = self.scalers[param].fit_transform(features)
            y = historical_data[param].values
            self.models[param].fit(X, y)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Forward fill and then backward fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df

    def predict(self, location: str, target_date: str) -> dict:
        # Get recent data for prediction
        lat, lon = 19.0760, 72.8777  # Mumbai's coordinates
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        recent_data = self.data_fetcher.fetch_historical_data(lat, lon, start_date, end_date)
        
        recent_data = self._clean_data(recent_data)
        features = self._prepare_features(recent_data)
        
        # Make predictions
        predictions = {}
        for param in self.models.keys():
            X = self.scalers[param].transform(features[-1:])
            pred = self.models[param].predict(X)[0]
            predictions[param] = round(float(pred), 2)
        
        return {
            "temperature": f"{predictions['temperature']}°C",
            "rain_probability": f"{min(100, max(0, predictions['precipitation'] * 100))}%",
            "humidity": f"{min(100, max(0, predictions['humidity']))}%",
            "wind_speed": f"{predictions['wind_speed']} km/h",
            "explanation": self._generate_explanation(predictions)
        }

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        features = pd.DataFrame({
            'hour': df['timestamp'].dt.hour,
            'day': df['timestamp'].dt.day,
            'month': df['timestamp'].dt.month,
            'temperature': df['temperature'],
            'humidity': df['humidity'],
            'wind_speed': df['wind_speed']
        })
        return features.values

    def _generate_explanation(self, predictions: dict) -> str:
        if predictions['precipitation'] > 0.5:
            weather_type = "heavy rainfall"
        elif predictions['precipitation'] > 0.2:
            weather_type = "moderate rainfall"
        elif predictions['precipitation'] > 0:
            weather_type = "light rainfall"
        else:
            weather_type = "clear weather"
            
        return f"Based on real-time weather data, expect {weather_type} with a temperature of {predictions['temperature']}°C" 