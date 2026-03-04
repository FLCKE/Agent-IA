"""
Labo 4 — Orchestration Hybride (Planner + CrewAI + LangGraph) avec Ollama.
"""

from dataclasses import dataclass
from typing import Dict, List

from ollama_utils import CHAT_MODEL, check_chat_ready, ollama_generate


def planner(goal: str, ollama_enabled: bool) -> List[str]:
    """Planner piloté par Gemma (ou fallback déterministe)."""
    print(f"[PLANNER] Objectif reçu : {goal}")
    if not ollama_enabled:
        return ["Collecte", "Synthèse", "Validation"]

    prompt = f"""
Tu es un planner d'orchestration agentique.
Objectif: {goal}

Donne uniquement 3 étapes, une par ligne, parmi:
Collecte
Synthèse
Validation
""".strip()
    answer = ollama_generate(prompt, temperature=0.0)
    steps = [s.strip("-• \t") for s in answer.splitlines() if s.strip()]
    kept = [s for s in steps if s in {"Collecte", "Synthèse", "Validation"}]
    return kept if kept else ["Collecte", "Synthèse", "Validation"]


@dataclass
class Agent:
    role: str
    goal: str

    def run(self, state: Dict, ollama_enabled: bool) -> Dict:
        if self.role == "Collector":
            if not ollama_enabled:
                state["data"] = [
                    "Les investissements dans le solaire augmentent en Europe.",
                    "Le stockage batterie devient un levier stratégique.",
                    "Les politiques publiques accélèrent l'adoption des ENR.",
                ]
            else:
                prompt = (
                    "Donne 3 faits courts sur les tendances récentes de l'énergie verte, "
                    "une ligne par fait, en français."
                )
                facts = [l.strip("-• \t") for l in ollama_generate(prompt, temperature=0.2).splitlines() if l.strip()]
                state["data"] = facts[:3] if facts else ["Données insuffisantes"]
        elif self.role == "Synthesizer":
            items = state.get("data", [])
            if not ollama_enabled:
                state["draft"] = (
                    "Synthèse: "
                    + " | ".join(items[:2])
                    + " | Tendance générale : croissance continue de l'énergie verte."
                )
            else:
                prompt = f"""
Tu es un agent de synthèse.
Résume en 4 phrases max pour un décideur.
Faits: {items}
""".strip()
                state["draft"] = ollama_generate(prompt, temperature=0.2)
        elif self.role == "Validator":
            draft = state.get("draft", "")
            if not ollama_enabled:
                state["validated"] = True if len(draft) > 40 else False
                state["final_note"] = draft + "\nValidation: OK" if state["validated"] else "Validation: ÉCHEC"
            else:
                prompt = f"""
Tu es un validateur qualité.
Texte: {draft}

Réponds strictement sur 2 lignes:
Validation: OK ou KO
Commentaire: ...
""".strip()
                verdict = ollama_generate(prompt, temperature=0.0)
                is_ok = "Validation: OK" in verdict
                state["validated"] = is_ok
                state["final_note"] = draft + "\n" + verdict
        return state


def run_state_graph(goal: str, ollama_enabled: bool) -> Dict:
    """Exécution d'un mini graphe d'états (inspiré LangGraph)."""
    plan = planner(goal, ollama_enabled)
    state = {"goal": goal, "plan": plan, "current_step": None}

    crew = {
        "Collecte": Agent(role="Collector", goal="Collecter des données fiables"),
        "Synthèse": Agent(role="Synthesizer", goal="Produire une synthèse claire"),
        "Validation": Agent(role="Validator", goal="Vérifier qualité et cohérence"),
    }

    for step in plan:
        print(f"[ETAPE] {step}")
        state["current_step"] = step
        state = crew[step].run(state, ollama_enabled)

    return state


def governance_graph_mermaid() -> str:
    return """
flowchart LR
    A[Goal] --> B[Planner]
    B --> C[State: Collecte]
    C --> D[Collector Agent]
    D --> E[State: Synthèse]
    E --> F[Synthesizer Agent]
    F --> G[State: Validation]
    G --> H[Validator Agent]
    H --> I[Final Note]
""".strip()


if __name__ == "__main__":
    objective = "Rédiger une note de synthèse sur l’énergie verte."
    print("=== LABO 4 — ORCHESTRATION HYBRIDE ===")
    ok, message = check_chat_ready()
    print(f"[CHECK] {message} | Modèle: {CHAT_MODEL}")

    final_state = run_state_graph(objective, ollama_enabled=ok)

    print("\n=== LIVRABLE FINAL ===")
    print(final_state.get("final_note", "Aucun livrable"))

    print("\n=== DIAGRAMME DE GOUVERNANCE (MERMAID) ===")
    print(governance_graph_mermaid())
