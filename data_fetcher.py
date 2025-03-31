import requests
import pandas as pd
from datetime import datetime, timedelta

class OpenMeteoFetcher:
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    def fetch_historical_data(self, latitude: float, longitude: float, start_date: str, end_date: str):
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m",
                      "wind_speed_10m", "surface_pressure"]
        }
        
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception("Failed to fetch data from OpenMeteo")
            
        data = response.json()
        
        # Convert to pandas DataFrame
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["hourly"]["time"]),
            "temperature": data["hourly"]["temperature_2m"],
            "precipitation": data["hourly"]["precipitation"],
            "humidity": data["hourly"]["relative_humidity_2m"],
            "wind_speed": data["hourly"]["wind_speed_10m"],
            "pressure": data["hourly"]["surface_pressure"]
        })
        
        return df 