import sys

def human_approval(action: str) -> bool:
    """Simule une demande d'approbation humaine."""
    print(f"DEMANDE D'APPROBATION] L'agent veut : {action}")
    # En environnement non interactif, on simule l'acceptation
    # print("Appuyez sur Entrée pour approuver (ou tapez 'non')...")
    # choice = input().lower()
    choice = "oui" # Simulation automatique pour le lab
    return choice != "non"

def run_guarded_agent(goal: str, max_steps: int = 5):
    print(f"=== Objectif : {goal} (Limite : {max_steps} étapes) ===")
    
    current_step = 0
    token_cost = 0

    while current_step < max_steps:
        current_step += 1
        print(f"🚀 Étape {current_step}")
        
        # Simulation d'une action risquée
        if current_step == 3:
            if not human_approval("Supprimer les fichiers temporaires du système"):
                print("⚠️ Action refusée par l'humain. L'agent change de stratégie.")
                continue
            else:
                print("✅ Action approuvée et exécutée.")

        # Simulation du coût des tokens
        token_cost += 150 
        print(f"  Statut : En cours... (Tokens utilisés : {token_cost})")

        if current_step == max_steps - 1:
            print("⚠️ ARRÊT DE SÉCURITÉ : Limite d'itérations bientôt atteinte.")
            break

    print(f"=== FIN DE L'EXÉCUTION (Total étapes: {current_step}) ===")

if __name__ == "__main__":
    run_guarded_agent("Nettoyer le disque dur de manière autonome")
