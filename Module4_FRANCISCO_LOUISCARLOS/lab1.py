def get_weather(city):
    return f"Il fait 23°C à {city}"

# Simulation du prompt utilisateur
user_prompt = "Quelle est la météo à Paris ?"

# Simulation de la décision de l'agent
city = "Paris"
response = get_weather(city)

print(response)
