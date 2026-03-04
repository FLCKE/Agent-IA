import subprocess
import json
from typing import List

MODEL = "gemma3:1b"

def ollama_chat(prompt: str, model: str = MODEL) -> str:
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    return result.stdout.strip()

def search_news(topic: str) -> str:
    print(f"  [ACTION] Recherche d'articles sur : {topic}")
    return f"Données brutes : Article 1 sur {topic}, Article 2, Article 3."

def summarize_data(data: str) -> str:
    print(f"  [ACTION] Résumé des données...")
    prompt = f"Résume ces données en 3 points clés : {data}"
    return ollama_chat(prompt)

def write_report(summary: str) -> str:
    print(f"  [ACTION] Rédaction du rapport final...")
    prompt = f"Rédige un court rapport structuré à partir de ce résumé : {summary}"
    return ollama_chat(prompt)

def autonomous_planner(goal: str):
    print(f"=== Objectif : {goal} ===")
    
    # Étape 1 : Planification
    print("[PLANIFICATION] L'agent génère son plan...")
    prompt_plan = f"""
    Ton objectif est : '{goal}'.
    Donne une liste JSON d'étapes (3 maximum) pour atteindre cet objectif.
    Format attendu : ["étape 1", "étape 2", "étape 3"]
    """
    plan_str = ollama_chat(prompt_plan)
    try:
        plan = json.loads(plan_str.strip("`").replace("json", "").strip())
    except:
        plan = ["Rechercher", "Résumer", "Rédiger"] # Plan de secours
    
    print(f"Plan généré : {plan}")

    # Étape 2 : Exécution dynamique
    results = {}
    data = search_news(goal)
    summary = summarize_data(data)
    final_report = write_report(summary)

    print("=== LIVRABLE FINAL ===")
    print(final_report)

if __name__ == "__main__":
    autonomous_planner("Les 3 dernières actualités sur l'IA générative")