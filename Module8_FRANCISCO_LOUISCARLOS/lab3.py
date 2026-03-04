"""
Labo 3 — Créer un Agent Hybride (RAG + Multi-Agent) avec Ollama + Gemma.

Prérequis:
  - ollama serve
  - ollama pull gemma3:1b
  - ollama pull nomic-embed-text
"""

from dataclasses import dataclass
import math
from typing import List

from ollama_utils import (
    CHAT_MODEL,
    EMBED_MODEL,
    check_chat_ready,
    check_embedding_ready,
    ollama_embedding,
    ollama_generate,
)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SimpleVectorIndex:
    """Index vectoriel local basé sur embeddings Ollama."""

    def __init__(self, texts: List[str], vectors: List[List[float]], use_embeddings: bool = True):
        self.texts = texts
        self.vectors = vectors
        self.use_embeddings = use_embeddings

    @classmethod
    def from_texts(cls, texts: List[str], use_embeddings: bool = True) -> "SimpleVectorIndex":
        if use_embeddings:
            vectors = [ollama_embedding(text) for text in texts]
            return cls(texts, vectors, use_embeddings=True)
        return cls(texts, [], use_embeddings=False)

    @staticmethod
    def _lexical_score(question: str, text: str) -> float:
        q_words = {w.strip("\n.,;:!?'\"").lower() for w in question.split() if w.strip()}
        t_words = {w.strip("\n.,;:!?'\"").lower() for w in text.split() if w.strip()}
        if not q_words:
            return 0.0
        return len(q_words & t_words) / len(q_words)

    def query(self, question: str, top_k: int = 2) -> List[str]:
        scored = []
        if self.use_embeddings:
            q_vec = ollama_embedding(question)
            for txt, vec in zip(self.texts, self.vectors):
                scored.append((cosine_similarity(q_vec, vec), txt))
        else:
            for txt in self.texts:
                scored.append((self._lexical_score(question, txt), txt))
        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        return [txt for _, txt in ranked[:top_k]]


@dataclass
class Agent:
    role: str
    goal: str

    def act(self, task: str, context: str) -> str:
        prompt = f"""
Tu es l'agent {self.role}.
Objectif: {self.goal}
Mission globale: {task}
Contexte RAG: {context}

Réponds de manière concise, structurée et en français.
""".strip()
        return ollama_generate(prompt)


class Crew:
    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def kickoff(self, task: str, retrieved_context: str) -> str:
        logs = [f"Mission: {task}"]
        intermediate = retrieved_context
        for agent in self.agents:
            step_output = agent.act(task, intermediate)
            logs.append(f"[{agent.role}] {step_output}")
            intermediate = step_output
        return "\n".join(logs)


def mermaid_flow() -> str:
    return """
flowchart TD
    A[User Goal] --> B[RAG Query\nLlamaIndex-like]
    B --> C[Top Documents]
    C --> D[Researcher Agent\nCrewAI-like]
    D --> E[Writer Agent\nCrewAI-like]
    E --> F[Final Answer]
""".strip()


if __name__ == "__main__":
    try:
        print("=== LABO 3 — AGENT HYBRIDE RAG + MULTI-AGENT (OLLAMA) ===")
        chat_ok, chat_message = check_chat_ready()
        emb_ok, emb_message = check_embedding_ready()
        print(f"[CHECK-CHAT] {chat_message} | Chat: {CHAT_MODEL}")
        print(f"[CHECK-EMBED] {emb_message} | Embedding: {EMBED_MODEL}")
        if not chat_ok:
            raise RuntimeError("Ollama n'est pas prêt pour le modèle chat")

        docs = [
            "En 2025, l'IA générative devient multimodale avec texte, image et audio.",
            "Les agents IA collaboratifs automatisent la recherche et la rédaction de rapports.",
            "La gouvernance IA impose traçabilité, auditabilité et supervision humaine.",
            "Les entreprises adoptent des assistants IA internes pour accélérer la prise de décision.",
        ]
        if emb_ok:
            index = SimpleVectorIndex.from_texts(docs, use_embeddings=True)
        else:
            print("[MODE FALLBACK] Embeddings indisponibles, passage en retrieval lexical.")
            index = SimpleVectorIndex.from_texts(docs, use_embeddings=False)

        user_task = "Analyse des tendances IA 2025"
        retrieved = index.query(user_task, top_k=2)
        context = " | ".join(retrieved)

        researcher = Agent(role="Researcher", goal="Analyser les tendances à partir du contexte")
        writer = Agent(role="Writer", goal="Produire une synthèse exploitable pour un décideur")
        crew = Crew(agents=[researcher, writer])
        result = crew.kickoff(user_task, context)

        print(result)
        print("\n=== SCHÉMA D'ORCHESTRATION (MERMAID) ===")
        print(mermaid_flow())

    except Exception as exc:
        print("[ERREUR] Exécution impossible avec Ollama.")
        print(f"Détail: {exc}")
        print("\nVérifie:")
        print("1) ollama serve")
        print(f"2) ollama pull {CHAT_MODEL}")
        print(f"3) ollama pull {EMBED_MODEL}")
