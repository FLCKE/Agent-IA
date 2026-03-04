import subprocess
from datetime import datetime

MODEL = "gemma3:1b"

def call_ollama(prompt: str, model: str = MODEL) -> str:
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",      # ✅ force UTF-8
        errors="replace"       # ✅ évite crash si caractère bizarre
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ollama error: {result.stderr.strip()}")
    return result.stdout.strip()

# ---------------------------
# 1) PLANNER (penser)
# ---------------------------
def planner(goal: str) -> list[str]:
    # Simple (comme l'énoncé) mais adaptable selon le goal
    return ["Chercher info", "Résumer", "Préparer email"]

# ---------------------------
# Outils (agir)
# ---------------------------
def tool_search_news() -> list[dict]:
    """
    Outil de recherche simulé.
    Dans un vrai projet: API news / RSS / web scraping.
    """
    return [
        {"title": "IA : nouvelles avancées sur les modèles compacts", "source": "TechDaily", "date": "2025-12-20"},
        {"title": "L’Europe discute de nouvelles règles pour l’IA", "source": "EuroNews", "date": "2025-12-19"},
        {"title": "Cybersécurité : hausse des attaques phishing", "source": "SecurityNow", "date": "2025-12-18"},
    ]

def tool_summarize_news(articles: list[dict]) -> str:
    articles_text = "\n".join([f"- {a['date']} | {a['source']} | {a['title']}" for a in articles])
    prompt = (
        "Tu es un assistant qui résume des actualités.\n"
        "Fais un résumé en 5 lignes max, en français, ton neutre.\n\n"
        "Articles:\n"
        f"{articles_text}\n\n"
        "Résumé:"
    )
    return call_ollama(prompt)

def tool_prepare_email(summary: str) -> str:
    today = datetime.now().strftime("%d/%m/%Y")
    subject = f"Résumé des actualités — {today}"
    body = (
        "Bonjour,\n\n"
        "Voici le résumé des actualités :\n"
        f"{summary}\n\n"
        "Bonne journée,\n"
        "Louis-Carlos"
    )
    email = f"Objet: {subject}\n\n{body}"
    return email

# ---------------------------
# 2) EXECUTOR (agir)
# ---------------------------
def executor(plan: list[str]) -> dict:
    state = {}  # mémoire de travail entre étapes

    for step in plan:
        print("\n🧩 Étape :", step)

        if step == "Chercher info":
            articles = tool_search_news()
            state["articles"] = articles
            print("👀 Observation: articles trouvés =", len(articles))
            for a in articles:
                print(f"   • {a['date']} — {a['source']} — {a['title']}")

        elif step == "Résumer":
            summary = tool_summarize_news(state["articles"])
            state["summary"] = summary
            print("👀 Observation: résumé généré")
            print(summary)

        elif step == "Préparer email":
            email = tool_prepare_email(state["summary"])
            state["email"] = email
            print("👀 Observation: email prêt à envoyer")
            print("\n" + email)

        else:
            print("⚠️ Étape inconnue:", step)

    return state

# ---------------------------
# 3) COMBINE
# ---------------------------
goal = "Préparer un résumé des actualités et l’envoyer"
print("🎯 Goal:", goal)

plan = planner(goal)
print("\n🗺️ Plan:")
for i, s in enumerate(plan, 1):
    print(f"  {i}. {s}")

executor(plan)
