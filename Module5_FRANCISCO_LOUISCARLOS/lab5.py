import subprocess

MODEL = "gemma3:1b"

# =========================
# Utilitaire Ollama
# =========================
def call_ollama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", MODEL],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace"
    )
    return result.stdout.strip()

# =====================================================
# PARTIE 1 — LangGraph (raisonnement en graphe)
# =====================================================
class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = {}

    def add_nodes(self, nodes):
        self.nodes.extend(nodes)

    def set_edges(self, edges):
        self.edges = edges

    def run(self, start_node):
        current = start_node
        print("🧠 Exécution du graphe logique (LangGraph simulé)\n")

        while current:
            print(f"🔷 Noeud : {current}")

            prompt = (
                f"Tu es un agent IA.\n"
                f"Rôle actuel : {current}\n"
                f"Objectif : créer un rapport sur la cybersécurité.\n"
                f"Explique brièvement ce que tu fais à cette étape."
            )
            output = call_ollama(prompt)
            print("   ⚙️ Action / Observation :", output, "\n")

            if current not in self.edges:
                print("✅ Réponse finale produite.")
                break

            current = self.edges[current]

# Création du graphe
graph = Graph()
graph.add_nodes(["Recherche", "Synthèse", "Vérification", "Réponse"])
graph.set_edges({
    "Recherche": "Synthèse",
    "Synthèse": "Vérification",
    "Vérification": "Réponse"
})

# =====================================================
# PARTIE 2 — Semantic Kernel Planner (conceptuel)
# =====================================================
class Plan:
    def __init__(self, steps):
        self.steps = steps

class SKPlanner:
    def create_plan(self, goal: str) -> Plan:
        prompt = (
            "Tu es un planner IA (Semantic Kernel).\n"
            f"Objectif : {goal}\n"
            "Découpe cet objectif en étapes logiques numérotées.\n"
            "Réponds uniquement par une liste d'étapes."
        )
        plan_text = call_ollama(prompt)

        steps = [
            line.strip("-•0123456789. ")
            for line in plan_text.splitlines()
            if line.strip()
        ]
        return Plan(steps)

# =========================
# EXECUTION
# =========================
if __name__ == "__main__":
    print("\n==============================")
    print("LABO 5 — LangGraph")
    print("==============================\n")
    graph.run(start_node="Recherche")

    print("\n==============================")
    print("LABO 5 — Semantic Kernel Planner")
    print("==============================\n")

    goal = "Créer un rapport sur la cybersécurité."
    planner = SKPlanner()
    plan = planner.create_plan(goal)

    print("🎯 Goal :", goal)
    print("\n🗺️ Plan généré :")
    for i, step in enumerate(plan.steps, 1):
        print(f"{i}. {step}")
