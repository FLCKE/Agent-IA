from tenacity import retry, stop_after_attempt, wait_fixed
import time


# -----------------------------
# 1) Fonction API simulée (échec)
# -----------------------------
@retry(
    stop=stop_after_attempt(3),   # 3 tentatives
    wait=wait_fixed(2),           # 2 secondes entre chaque tentative
    reraise=True                  # relancer l'exception après échec
)
def get_weather(city: str) -> str:
    print(f"🌐 Tentative API météo pour {city}...")
    time.sleep(0.5)  # simulation délai réseau
    raise Exception("Erreur API météo (clé invalide ou timeout)")


# -----------------------------
# 2) Fallback
# -----------------------------
def fallback_weather(city: str) -> str:
    return (
        f"⚠️ Impossible d'obtenir la météo pour {city} pour le moment. "
        "Merci de réessayer plus tard."
    )


# -----------------------------
# 3) Agent résilient
# -----------------------------
def weather_agent(city: str) -> str:
    try:
        result = get_weather(city)
        return result
    except Exception as e:
        print(f"❌ Échec après retries : {e}")
        return fallback_weather(city)


# -----------------------------
# 4) Démo (livrable)
# -----------------------------
if __name__ == "__main__":
    print("👤 Utilisateur : Quelle est la météo à Paris ?\n")
    response = weather_agent("Paris")
    print("\n🤖 Agent :", response)
