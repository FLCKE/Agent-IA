from typing import Dict, List

def calculate_kpi(runs: List[Dict]) -> Dict:
    total_runs = len(runs)
    successes = sum(1 for r in runs if r["success"])
    errors = sum(1 for r in runs if r["critical_error"])
    total_requests = sum(r["api_calls"] for r in runs)
    total_steps = sum(r["steps"] for r in runs)

    return {
        "Succès (%)": (successes / total_runs) * 100,
        "Sécurité (Erreurs)": errors,
        "Coût (Moyen API)": total_requests / total_runs,
        "Efficacité (Étapes/Succès)": total_steps / (successes if successes > 0 else 1)
    }

if __name__ == "__main__":
    # Simulation de 5 exécutions de l'agent
    agent_runs = [
        {"success": True, "critical_error": False, "api_calls": 5, "steps": 3},
        {"success": True, "critical_error": False, "api_calls": 7, "steps": 4},
        {"success": False, "critical_error": False, "api_calls": 10, "steps": 10}, # Echec mais pas d'erreur critique
        {"success": True, "critical_error": False, "api_calls": 4, "steps": 2},
        {"success": False, "critical_error": True, "api_calls": 3, "steps": 3},  # Erreur critique
    ]

    kpis = calculate_kpi(agent_runs)

    print("=== TABLEAU D'ÉVALUATION DE L'AUTONOMIE ===")
    print("| Critère              | Mesure                 |")
    print("|----------------------|------------------------|")
    for k, v in kpis.items():
        print(f"| {k:<20} | {v:<22.2f} |")

    print("--- ANALYSE DE PERFORMANCE ---")
    print(f"L'agent affiche un taux de réussite de {kpis['Succès (%)']}%.")
    if kpis["Sécurité (Erreurs)"] > 0:
        print(f"⚠️ Alerte : {kpis['Sécurité (Erreurs)']} erreur(s) critique(s) détectée(s).")
    print(f"L'efficacité moyenne est de {kpis['Efficacité (Étapes/Succès)']} étapes par succès.")
