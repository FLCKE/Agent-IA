import subprocess
from typing import Dict

MODEL = "gemma3:1b"  # adapte si ton modèle s'appelle autrement dans `ollama list`

def ollama_chat(prompt: str, model: str = MODEL) -> str:
    """
    Appelle Ollama via CLI: `ollama run <model> "<prompt>"`
    Retourne la sortie texte du modèle.
    """
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ollama error: {result.stderr.strip()}")
    return result.stdout.strip()

def researcher(task: str) -> str:
    prompt = f"""
Tu es un agent 'Chercheur'. Ta mission: collecter des informations BRUTES, factuelles et structurées sur le sujet.
Ne fais pas de résumé, donne des points clés, définitions, exemples, mots-clés.
Sujet: {task}

Format attendu:
- Points clés (5 à 10 puces)
- Définitions courtes
- Exemples concrets
"""
    return ollama_chat(prompt)

def writer(data: str) -> str:
    prompt = f"""
Tu es un agent 'Rédacteur'. Ta mission: transformer les données brutes en synthèse claire et pédagogique.
Rédige un texte court (120 à 180 mots), structuré, sans jargon inutile.
Données brutes:
{data}
"""
    return ollama_chat(prompt)

def reviewer(text: str) -> str:
    prompt = f"""
Tu es un agent 'Relecteur'. Ta mission: critiquer la clarté et améliorer le texte.
1) Liste 3 améliorations concrètes (puces)
2) Fournis une version améliorée du texte (même longueur)
Texte:
{text}
"""
    return ollama_chat(prompt)

def run_pipeline(task: str) -> Dict[str, str]:
    print("=== Collaboration multi-agents (Ollama + Gemma 3) ===")
    print(f"Tâche: {task}\n")

    data = researcher(task)
    print("[Chercheur] -> Données brutes:\n", data, "\n")

    draft = writer(data)
    print("[Rédacteur] -> Synthèse:\n", draft, "\n")

    improved = reviewer(draft)
    print("[Relecteur] -> Améliorations + version finale:\n", improved, "\n")

    return {"research": data, "draft": draft, "final": improved}

if __name__ == "__main__":
    run_pipeline("les stratégies de déploiement CI/CD (Blue-Green, Canary, Rolling)")