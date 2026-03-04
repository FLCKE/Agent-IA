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

def autonomy_loop(mission: str):
    current_result = "Aucun résultat initial."
    print(f"=== Mission : {mission} ===")

    for i in range(3):
        print(f"🧭 ITÉRATION {i+1}")
        
        # 1. Action / Production
        prompt_action = f"""
        Mission : '{mission}'.
        Résultat précédent : {current_result}
        Produis une réponse améliorée à cette mission.
        """
        current_result = ollama_chat(prompt_action)
        print(f"  [ACT] Nouveau résultat produit.")

        # 2. Réflexion
        prompt_reflection = f"""
        Analyse ce résultat : '{current_result}'.
        Donne un point fort et un point à améliorer.
        Réponse courte attendue.
        """
        reflection = ollama_chat(prompt_reflection)
        print(f"  [REFLECT] Réflexion de l'agent :{reflection}")
        
        print("-" * 30)

    print("=== LIVRABLE FINAL APRÈS RÉFLEXION ===")
    print(current_result)

if __name__ == "__main__":
    autonomy_loop("Définir brièvement l'agent autonome en IA.")
