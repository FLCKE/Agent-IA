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

def self_check_agent(question: str):
    print(f"=== Question : {question} ===")

    # 1. Génération primaire
    print("[1] AGENT PRIMAIRE : Génération de la réponse...")
    primary_answer = ollama_chat(f"Réponds précisément à cette question : {question}")
    print(f"Réponse primaire :{primary_answer}")

    # 2. Vérification secondaire (Critique factuelle)
    print("[2] AGENT VÉRIFICATEUR : Fact-checking...")
    prompt_checker = f"""
    Voici une réponse : '{primary_answer}'.
    Vérifie les faits. Si des erreurs sont présentes, liste-les. Sinon, confirme.
    Question initiale : {question}
    """
    check_result = ollama_chat(prompt_checker)
    print(f"Rapport de vérification :{check_result}")

    # 3. Synthèse finale vérifiée
    print("[3] RÉPONSE FINALE VÉRIFIÉE...")
    prompt_final = f"""
    Synthétise la réponse finale en tenant compte de la vérification.
    Réponse primaire : {primary_answer}
    Rapport de vérification : {check_result}
    """
    verified_answer = ollama_chat(prompt_final)
    
    print("=== LIVRABLE FINAL VÉRIFIÉ ===")
    print(verified_answer)

if __name__ == "__main__":
    self_check_agent("Quelle est la date d'adoption du RGPD ?")
