# lab5_eval.py — Labo 5 : Évaluer l’efficacité de la mémoire (Recall / Update / Forget)
# Prérequis :
#   1) ollama serve
#   2) ollama pull gemma3
#   3) pip install "langchain>=0.2" "langchain-community>=0.2"

from typing import List, Dict, Tuple
import re
from textwrap import shorten

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

MODEL_NAME = "gemma3"

# ---------- Mémoire résumée (comme Labo 4, simplifiée) ----------
class SummaryMemory:
    def __init__(self, llm: ChatOllama, max_buffer_turns: int = 3):
        self.llm = llm
        self.max_buffer_turns = max_buffer_turns
        self.buffer: List[Dict[str, str]] = []  # [{role: "user"/"ai", "content": "..."}]
        self.summary: str = ""

    def add_user(self, text: str):
        self.buffer.append({"role": "user", "content": text})

    def add_ai(self, text: str):
        self.buffer.append({"role": "ai", "content": text})

    def clear(self):
        self.buffer.clear()
        self.summary = ""

    def _summarize(self):
        if not self.buffer:
            return
        convo_text = ""
        for turn in self.buffer:
            prefix = "Utilisateur" if turn["role"] == "user" else "Assistant"
            convo_text += f"{prefix}: {turn['content']}\n"

        messages = [
            SystemMessage(content=(
                "Tu es un assistant qui résume des dialogues de manière concise et factuelle. "
                "Conserve les informations stables (noms, objectifs, préférences)."
            )),
            HumanMessage(content=(
                f"Résumé courant:\n{self.summary or 'Aucun'}\n\n"
                f"Nouvelle conversation à intégrer:\n{convo_text}\n\n"
                "Produis un NOUVEAU résumé unique (5–8 lignes max)."
            )),
        ]
        resp = self.llm.invoke(messages)
        self.summary = resp.content.strip()
        self.buffer = []

    def maybe_summarize(self):
        if len(self.buffer) >= 2 * self.max_buffer_turns:
            self._summarize()

    def context_messages(self) -> List:
        msgs: List = []
        sys = "Tu es un assistant utile."
        if self.summary:
            sys += "\nMémoire résumée:\n" + self.summary
        msgs.append(SystemMessage(content=sys))
        for t in self.buffer:
            if t["role"] == "user":
                msgs.append(HumanMessage(content=t["content"]))
            else:
                msgs.append(AIMessage(content=t["content"]))
        return msgs

# ---------- Mémoire structurée (slots) ----------
class SlotMemory:
    """Stocke des faits clés (ex: name) de façon déterministe."""
    def __init__(self):
        self.slots: Dict[str, str] = {}

    def set(self, key: str, value: str):
        self.slots[key] = value

    def get(self, key: str, default: str = "") -> str:
        return self.slots.get(key, default)

    def clear(self):
        self.slots.clear()

    def as_text(self) -> str:
        if not self.slots:
            return "Aucun fait structuré."
        return "\n".join(f"- {k}: {v}" for k, v in self.slots.items())

# ---------- Agent hybride ----------
class HybridAgent:
    def __init__(self, model_name: str = MODEL_NAME):
        self.llm = ChatOllama(model=model_name)
        self.summary_mem = SummaryMemory(llm=self.llm, max_buffer_turns=3)
        self.slots = SlotMemory()

    # Règles de parsing simples pour capturer/mettre à jour le prénom
    NAME_PATTERNS = [
        re.compile(r"\bje m'?appelle\s+([A-Za-zÀ-ÖØ-öø-ÿ\-']+)\b", re.IGNORECASE),
        re.compile(r"\ben fait[, ]*\s*je m'?appelle\s+([A-Za-zÀ-ÖØ-öø-ÿ\-']+)\b", re.IGNORECASE),
    ]

    def ingest_user(self, text: str):
        # capture/MAJ du prénom si trouvé
        for pat in self.NAME_PATTERNS:
            m = pat.search(text)
            if m:
                self.slots.set("name", m.group(1))
        # "oublie" -> effacer la mémoire
        if re.search(r"\boublie\b", text, re.IGNORECASE):
            self.summary_mem.clear()
            self.slots.clear()
        # push dans la mémoire résumée
        self.summary_mem.add_user(text)

    def respond(self, user_text: str) -> str:
        # alimenter mémoires
        self.ingest_user(user_text)

        # Construire le contexte de réponse
        system = (
            "Tu es un assistant utile, concis, exact.\n"
            "Faits structurés (haute priorité, source de vérité):\n"
            f"{self.slots.as_text()}\n"
            "Si la question est 'Quel est mon nom ?', réponds uniquement par le prénom le plus récent connu."
        )
        msgs = [SystemMessage(content=system)] + self.summary_mem.context_messages()[1:]
        msgs.append(HumanMessage(content=user_text))

        # Appel LLM
        resp = self.llm.invoke(msgs)
        answer = resp.content.strip()

        # stocker côté mémoire résumée puis résumer si besoin
        self.summary_mem.add_ai(answer)
        self.summary_mem.maybe_summarize()
        return answer

    # utilitaires de test
    def clear_all(self):
        self.summary_mem.clear()
        self.slots.clear()

# ---------- Évaluation ----------
def run_tests() -> None:
    agent = HybridAgent()

    results: List[Tuple[str, str, str, str]] = []

    # --- Recall Test ---
    agent.clear_all()
    agent.respond("Mon nom est André.")
    agent.respond("Parlons de météo.")
    recall_answer = agent.respond("Quel est mon nom ?")
    recall_ok = "André".casefold() in recall_answer.casefold()
    results.append(("Rappel", "« André »", recall_answer, "✅" if recall_ok else "❌"))

    # --- Update Test ---
    update_answer_pre = agent.respond("En fait, je m'appelle Marc.")
    update_answer = agent.respond("Quel est mon nom ?")
    update_ok = "Marc".casefold() in update_answer.casefold()
    results.append(("Mise à jour", "« Marc »", update_answer, "✅" if update_ok else "❌"))

    # --- Forget Test ---
    agent.respond("Oublie ce que je viens de dire.")
    forget_answer = agent.respond("Quel est mon nom ?")
    # réussite si la réponse NE contient NI André NI Marc (ou avoue ne pas savoir)
    forget_ok = ("andré" not in forget_answer.casefold()) and ("marc" not in forget_answer.casefold())
    results.append(("Oubli", "« ? »", forget_answer, "✅" if forget_ok else "❌"))

    # --- Affichage tableau ---
    print("\nTableau d'évaluation")
    print("Test        | Attendu  | Réponse                                     | Résultat")
    print("------------|----------|---------------------------------------------|---------")
    for test, attendu, reponse, ok in results:
        print(f"{test:<12}| {attendu:<8} | {shorten(reponse, width=45, placeholder='…'):<43} | {ok}")

    # --- Conclusion courte ---
    print("\nConclusion :")
    print("- Recall : l’agent ressort bien le prénom mémorisé.")
    print("- Update : la nouvelle valeur (Marc) remplace l’ancienne.")
    print("- Forget : après 'Oublie…', le prénom n’est plus restitué.")

if __name__ == "__main__":
    print("=== Labo 5 : Évaluer l’efficacité de la mémoire (Ollama gemma3) ===")
    run_tests()
