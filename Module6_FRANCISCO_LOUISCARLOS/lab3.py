import subprocess
from typing import Dict

MODEL = "gemma3:1b"

def ollama_chat(prompt: str, model: str = MODEL) -> str:
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    if result.returncode != 0:
        return f"Erreur Ollama: {result.stderr.strip()}"
    return result.stdout.strip()

shared_memory = {}

def researcher_agent(task: str):
    print(f"[Chercheur] Traitement de la tâche: {task}")
    prompt = f"Tu es un Chercheur. Récupère des faits clés sur le sujet suivant: {task}."
    data = ollama_chat(prompt)
    shared_memory["data"] = data
    print("[Chercheur] Données enregistrées dans la mémoire partagée.")

def writer_agent():
    data = shared_memory.get("data", "Aucune donnée disponible.")
    print("[Rédacteur] Lecture des données de la mémoire partagée.")
    prompt = f"Tu es un Rédacteur. Utilise ces informations pour écrire un court résumé: {data}."
    summary = ollama_chat(prompt)
    shared_memory["summary"] = summary
    print("[Rédacteur] Résumé enregistré dans la mémoire partagée.")

def reviewer_agent():
    summary = shared_memory.get("summary", "Aucun résumé disponible.")
    print("[Relecteur] Lecture du résumé de la mémoire partagée.")
    prompt = f"Tu es un Relecteur. Critique et donne un feedback sur ce texte: {summary}."
    feedback = ollama_chat(prompt)
    shared_memory["feedback"] = feedback
    print("[Relecteur] Feedback enregistré dans la mémoire partagée.")

if __name__ == "__main__":
    task_subject = "L'essor de la robotique humanoïde en 2025"
    print(f"=== Début du Lab 3 : Mémoire Partagée pour '{task_subject}' ===\n")

    researcher_agent(task_subject)
    print("-" * 30)
    
    writer_agent()
    print("-" * 30)
    
    reviewer_agent()
    print("-" * 30)

    print("\n=== Contenu final de la mémoire partagée ===")
    for key, value in shared_memory.items():
        print(f"\n[{key.upper()}]:\n{value[:200]}...")
    
    print("\n--- FIN DU LAB 3 ---")
