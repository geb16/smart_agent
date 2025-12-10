import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


def tool_weather(city: str, temperature_unit: str = "celsius") -> str:
    """Get current weather for a city using OpenWeather with unit preference."""

    if not OPENWEATHER_API_KEY:
        return f"Weather API key not configured for {city}."

    try:
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}  # API returns Celsius
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()

        data = resp.json()
        temp_c = data["main"]["temp"]
        desc = data["weather"][0]["description"]

        # ---- Unit handling based on LTM preference ----
        if temperature_unit.lower() == "fahrenheit":
            temp = temp_c * 9 / 5 + 32
            unit = "°F"
        else:
            temp = temp_c
            unit = "°C"

        return f"{city}: {round(temp, 2)}{unit}, {desc}"

    except Exception as e:
        return f"Error getting weather for {city}: {str(e)}"
