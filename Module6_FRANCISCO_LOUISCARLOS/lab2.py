import subprocess
from typing import List, Callable, Dict

MODEL = "gemma3:1b"

def ollama_chat(prompt: str, model: str = MODEL) -> str:
    """
    Appelle Ollama via CLI: `ollama run <model> "<prompt>"`
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

def researcher(mission: str) -> str:
    prompt = f"Tu es un Chercheur. Collecte des informations clés sur: {mission}. Sois factuel."
    return f"[Chercheur]: {ollama_chat(prompt)}"

def writer(mission: str) -> str:
    prompt = f"Tu es un Rédacteur. Rédige un court paragraphe d'introduction sur: {mission}."
    return f"[Rédacteur]: {ollama_chat(prompt)}"

def reviewer(mission: str) -> str:
    prompt = f"Tu es un Relecteur. Analyse les points critiques de: {mission}."
    return f"[Relecteur]: {ollama_chat(prompt)}"

class Manager: 
    def __init__(self, workers: List[Callable[[str], str]]): 
        self.workers = workers 
        self.results = {}

    def run(self, mission: str): 
        print(f"--- Début de la Mission : {mission} ---") 
        for w in self.workers: 
            agent_name = w.__name__
            print(f"Délégation à l'agent '{agent_name}'...")
            result = w(mission) 
            self.results[agent_name] = result
            print(f"-> Réponse de {agent_name} reçue.\n") 

        print("--- Agrégation du résultat final ---")
        for agent, res in self.results.items():
            print(f"[{agent.upper()}]: {res[:100]}...")

if __name__ == "__main__":
    workers_list = [researcher, writer, reviewer]
    manager = Manager(workers_list) 
    manager.run("L'impact de l'IA générative sur le développement logiciel en 2025")
