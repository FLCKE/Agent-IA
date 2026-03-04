import subprocess

MODEL = "gemma3:1b"

def ollama_chat(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", MODEL, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    return result.stdout.strip()

def reflexive_agent(task: str):
    print(f"=== Tâche : {task} ===")

    # 1. Étape de génération initiale
    print("[1] Génération de la réponse initiale...")
    initial_answer = ollama_chat(f"Réponds à cette tâche : {task}")
    print(f"Réponse initiale :{initial_answer}")

    # 2. Étape d'auto-réflexion (Critique)
    print("[2] Auto-réflexion (Critique)...")
    prompt_critique = f"""
    Voici une réponse : '{initial_answer}'.
    Critique la précision factuelle et la clarté. Liste 2 points négatifs.
    """
    critique = ollama_chat(prompt_critique)
    print(f"Critique interne de l'agent :{critique}")

    # 3. Étape de correction
    print("[3] Auto-correction...")
    prompt_correction = f"""
    Corrige la réponse initiale : '{initial_answer}'.
    Tiens compte de cette critique : '{critique}'.
    Donne la version finale corrigée.
    """
    final_answer = ollama_chat(prompt_correction)
    
    print("=== LIVRABLE FINAL CORRIGÉ ===")
    print(final_answer)

if __name__ == "__main__":
    reflexive_agent("Explique le fonctionnement du RGPD en une phrase.")
