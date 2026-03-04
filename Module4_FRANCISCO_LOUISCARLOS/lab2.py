import requests
import os

API_KEY = os.getenv("OPENWEATHER_KEY")

def get_weather(city):
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric&lang=fr"
    )
    response = requests.get(url, timeout=10)
    data = response.json()

    # Gestion d'erreur (si la ville n'existe pas ou clé invalide, etc.)
    if response.status_code != 200:
        return f"Erreur API : {response.status_code} {data.get('message', 'Erreur inconnue')}"

    description = data["weather"][0]["description"]
    temperature = data["main"]["temp"]

    return f"{description} et {temperature}°C à {city}"

print(get_weather("Paris"))
