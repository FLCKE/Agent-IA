"""
Labo 2 — Associer Patterns et Frameworks (version Ollama + Gemma)
Objectif : produire automatiquement la matrice de choix.
"""

from ollama_utils import CHAT_MODEL, check_chat_ready, ollama_generate


def build_prompt() -> str:
    return """
Tu es architecte IA senior.

Tâche 1:
Proposer un tableau markdown à 3 colonnes:
| Pattern | Framework idéal | Pourquoi |
avec exactement ces patterns:
- RAG
- Planner
- Multi-Agent

Tâche 2:
Proposer ensuite une combinaison hybride originale en 3 lignes:
Combinaison:
Principe:
Cas d'usage:

Contraintes:
- Réponse en français
- Réponse concise
""".strip()


def fallback_output() -> str:
    return """| Pattern | Framework idéal | Pourquoi |
|---|---|---|
| RAG | LlamaIndex | Ingestion documentaire et retrieval sémantique robuste. |
| Planner | Semantic Kernel | Planification orientée compétences et plugins. |
| Multi-Agent | CrewAI | Coordination claire entre rôles d'agents. |

Combinaison: LangGraph + LlamaIndex + CrewAI + Semantic Kernel
Principe: LangGraph orchestre les états, LlamaIndex fournit le contexte, CrewAI répartit l'exécution et Semantic Kernel pilote les outils.
Cas d'usage: Assistant de veille stratégique qui collecte, analyse et valide un briefing hebdomadaire."""


if __name__ == "__main__":
    print("=== LABO 2 — ASSOCIATION PATTERNS & FRAMEWORKS (OLLAMA) ===")
    ok, message = check_chat_ready()
    print(f"[CHECK] {message} | Modèle: {CHAT_MODEL}")

    if not ok:
        print("\n[MODE FALLBACK] Utilisation d'un rendu local.")
        print(fallback_output())
    else:
        print(ollama_generate(build_prompt(), temperature=0.1))
