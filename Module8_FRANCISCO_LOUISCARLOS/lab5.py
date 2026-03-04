"""
Labo 5 — Étude prospective (version Ollama + Gemma)
But : imaginer un agent du futur (2026-2027) avec génération locale.
"""

from ollama_utils import CHAT_MODEL, check_chat_ready, ollama_generate


def build_prompt() -> str:
    return """
Tu es un consultant IA senior.

Rédige une vision "Agent du futur (2026-2027)" en markdown avec exactement ces sections:
## 1) Architecture cible
## 2) Frameworks combinés
## 3) Gouvernance et fiabilité
## 4) Cas d'usage 2027

Contraintes:
- Réponse en français
- Style clair et professionnel
- 5 à 8 lignes par section
""".strip()


def fallback_vision() -> str:
    return """# Agent du futur (2026-2027) — Vision

## 1) Architecture cible
- **Noyau orchestrateur** basé sur un graphe d'états (LangGraph-like)
- **Mémoire hiérarchique** (court terme, long terme, mémoire d'épisodes)
- **RAG multimodal** (texte, image, audio, vidéo)
- **Routage multi-LLM** selon coût, latence, confidentialité et qualité
- **Couche de sécurité** : policy engine, garde-fous, validation humaine optionnelle

## 2) Frameworks combinés
- **LlamaIndex** : ingestion documentaire et retrieval avancé
- **CrewAI / AutoGen** : collaboration d'agents spécialisés (analyse, rédaction, QA)
- **LangGraph** : contrôle des transitions d'état et reprise sur erreur
- **Semantic Kernel** : plugins métiers, planification et connecteurs outils

## 3) Gouvernance et fiabilité
- Observabilité complète (traces, métriques, coûts, audits)
- Système d'auto-évaluation (agent critique + score de confiance)
- Vérification factuelle avant livraison (fact-check + citations)
- Gestion des risques : sandbox outils, contrôle d'accès, chiffrement des données

## 4) Cas d'usage 2027
Un "Chief-of-Staff AI" d'entreprise :
1. Comprend les objectifs hebdomadaires
2. Planifie les actions par équipes
3. Collecte et résume les données internes/externes
4. Rédige les livrables exécutifs
5. Auto-vérifie, justifie et propose des améliorations continues"""


if __name__ == "__main__":
    print("=== LABO 5 — ÉTUDE PROSPECTIVE ===")
    ok, message = check_chat_ready()
    print(f"[CHECK] {message} | Modèle: {CHAT_MODEL}")
    if ok:
        print(ollama_generate(build_prompt(), temperature=0.3))
    else:
        print("[MODE FALLBACK] Utilisation d'un rendu local.\n")
        print(fallback_vision())
