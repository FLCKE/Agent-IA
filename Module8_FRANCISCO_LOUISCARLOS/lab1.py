"""
Labo 1 — Identifier les Design Patterns (version Ollama + Gemma)
Objectif : classifier des use cases avec un LLM local.
"""

from ollama_utils import CHAT_MODEL, check_chat_ready, ollama_generate


USE_CASES = [
    "Chatbot FAQ",
    "Assistant personnel",
    "Chat PDF",
    "Équipe collaborative",
]


def build_prompt() -> str:
    use_cases = "\n".join(f"- {u}" for u in USE_CASES)
    return f"""
Tu es expert en design patterns d'agents IA.

Classifie chaque use case ci-dessous dans UN pattern principal parmi:
- Réflexe
- Mémoire
- RAG
- Multi-Agent

Use cases:
{use_cases}

Réponds uniquement en tableau markdown avec 3 colonnes:
| Use Case | Pattern | Justification |
""".strip()


def fallback_output() -> str:
    return """| Use Case | Pattern | Justification |
|---|---|---|
| Chatbot FAQ | Réflexe | Réponse directe sur intent simple et faible contexte. |
| Assistant personnel | Mémoire | Exploite les préférences et l'historique utilisateur. |
| Chat PDF | RAG | Recherche de passages puis génération avec sources. |
| Équipe collaborative | Multi-Agent | Répartition des tâches entre rôles spécialisés. |"""


if __name__ == "__main__":
    print("=== LABO 1 — IDENTIFICATION DES PATTERNS (OLLAMA) ===")
    ok, message = check_chat_ready()
    print(f"[CHECK] {message} | Modèle: {CHAT_MODEL}")

    if not ok:
        print("\n[MODE FALLBACK] Utilisation d'un rendu local.")
        print(fallback_output())
    else:
        response = ollama_generate(build_prompt(), temperature=0.1)
        print(response)
