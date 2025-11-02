# lab4.py — Mémoire résumée SANS langchain.memory
# Compatible avec LangChain ≥ 0.2.x + Ollama local

from typing import List, Dict
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

MODEL_NAME = "gemma3"

class SummaryMemory:
    def __init__(self, llm: ChatOllama, max_buffer_turns: int = 3):
        self.llm = llm
        self.max_buffer_turns = max_buffer_turns
        self.buffer: List[Dict[str, str]] = []
        self.summary: str = ""

    def add_user(self, text: str):
        self.buffer.append({"role": "user", "content": text})

    def add_ai(self, text: str):
        self.buffer.append({"role": "ai", "content": text})

    def _summarize(self):
        if not self.buffer:
            return
        convo_text = ""
        for turn in self.buffer:
            prefix = "Utilisateur" if turn["role"] == "user" else "Assistant"
            convo_text += f"{prefix}: {turn['content']}\n"

        messages = [
            SystemMessage(content=(
                "Tu es un assistant qui résume des conversations. "
                "Résumé court et factuel. Ne rajoute pas d'informations inventées."
            )),
            HumanMessage(content=(
                f"Résumé actuel:\n{self.summary or 'Aucun'}\n\n"
                f"Nouvel historique à intégrer:\n{convo_text}\n\n"
                "Donne un résumé unique, court (5-8 lignes)."
            ))
        ]

        resp = self.llm.invoke(messages)   # ✅ utilise invoke()
        self.summary = resp.content.strip()
        self.buffer = []  # on vide le buffer (on a condensé)

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


def main():
    llm = ChatOllama(model=MODEL_NAME)
    mem = SummaryMemory(llm=llm, max_buffer_turns=3)

    def ask(user_text: str):
        mem.add_user(user_text)
        msgs = mem.context_messages()
        resp = llm.invoke(msgs)   # ✅ ici aussi invoke()
        answer = resp.content.strip()
        print(answer)
        mem.add_ai(answer)
        mem.maybe_summarize()

    print("=== Démo Mémoire Résumée (gemma3 @ Ollama) ===")
    ask("Bonjour ! Je développe un agent d’IA pour Ydays. Le but est d’explorer différentes mémoires.")
    ask("Aujourd’hui je travaille sur la mémoire résumée pour économiser du contexte.")
    ask("Rappelle-toi que mon objectif est d’avoir de la cohérence sans exploser les tokens.")
    ask("Peux-tu me résumer ce que tu sais de mon projet ?")

    # Résumé final
    mem._summarize()
    print("\n--- Résumé interne de la mémoire ---")
    print(mem.summary)

if __name__ == "__main__":
    main()
