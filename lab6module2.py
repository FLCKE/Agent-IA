# lab6_persistence.py — Persistance JSON + rechargement (Ollama + gemma3)
# Prérequis :
#   1) ollama serve
#   2) ollama pull gemma3
#   3) pip install "langchain>=0.2" "langchain-community>=0.2"
#
# Utilisation (exemples) :
#   python lab6_persistence.py
#   -> tape : Je m'appelle André.
#   -> tape : save   (sauvegarde sur disque et affiche le JSON)
#   -> relance le script : tu verras le contenu rechargé, et il connaît déjà ton prénom

import json
import os
import sys
import tempfile
from typing import List, Dict
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

MODEL_NAME = "gemma3"
MEMORY_PATH = os.path.join(".", "memory.json")  # tu peux changer l’emplacement

# --------- Mémoire résumée minimaliste (comme Labo 4) ----------
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

    def to_dict(self) -> Dict:
        return {"summary": self.summary, "buffer": self.buffer}

    def load_dict(self, data: Dict):
        self.summary = data.get("summary", "")
        self.buffer = data.get("buffer", [])

    def _summarize(self):
        if not self.buffer:
            return
        convo_text = ""
        for turn in self.buffer:
            who = "Utilisateur" if turn["role"] == "user" else "Assistant"
            convo_text += f"{who}: {turn['content']}\n"

        messages = [
            SystemMessage(content=(
                "Tu es un assistant qui résume des dialogues de façon concise et factuelle. "
                "Conserve les informations stables (noms, objectifs, préférences)."
            )),
            HumanMessage(content=(
                f"Résumé courant:\n{self.summary or 'Aucun'}\n\n"
                f"Nouvel historique à intégrer:\n{convo_text}\n\n"
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

# --------- Mémoire structurée (slots) ----------
class SlotMemory:
    def __init__(self):
        self.slots: Dict[str, str] = {}

    def set(self, k: str, v: str):
        self.slots[k] = v

    def get(self, k: str, default: str = "") -> str:
        return self.slots.get(k, default)

    def clear(self):
        self.slots.clear()

    def to_dict(self) -> Dict:
        return dict(self.slots)

    def load_dict(self, data: Dict):
        self.slots = dict(data or {})

    def as_text(self) -> str:
        if not self.slots:
            return "Aucun fait structuré."
        return "\n".join(f"- {k}: {v}" for k, v in self.slots.items())

# --------- Gestion de la persistance JSON ----------
class PersistenceManager:
    def __init__(self, path: str):
        self.path = path

    def save(self, payload: Dict):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="mem_", suffix=".json", dir=os.path.dirname(self.path) or ".")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            # écriture atomique
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

    def load(self) -> Dict:
        if not os.path.exists(self.path):
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def delete(self):
        if os.path.exists(self.path):
            os.remove(self.path)

# --------- Agent hybride + persistance ----------
class Agent:
    def __init__(self, model_name: str = MODEL_NAME, memory_path: str = MEMORY_PATH):
        self.llm = ChatOllama(model=model_name)
        self.summary_mem = SummaryMemory(llm=self.llm, max_buffer_turns=3)
        self.slots = SlotMemory()
        self.store = PersistenceManager(memory_path)

        # au démarrage, tenter de charger
        data = self.store.load()
        self.summary_mem.load_dict(data.get("summary_mem", {}))
        self.slots.load_dict(data.get("slots", {}))

    def save(self):
        payload = {
            "summary_mem": self.summary_mem.to_dict(),
            "slots": self.slots.to_dict(),
        }
        self.store.save(payload)
        return payload

    def reset(self):
        self.summary_mem.summary = ""
        self.summary_mem.buffer = []
        self.slots.clear()
        self.store.delete()

    def respond(self, user_text: str) -> str:
        # capter un prénom si l’utilisateur dit “Je m’appelle X”
        low = user_text.lower()
        if "je m'appelle" in low or "je mappelle" in low:
            try:
                # dernier token comme prénom (simple pour la démo)
                name = user_text.split()[-1].strip(" .,!?:;\"'()[]")
                if name:
                    self.slots.set("name", name)
            except Exception:
                pass

        # construire contexte
        system = (
            "Tu es un assistant concis et exact.\n"
            "Faits structurés (source de vérité prioritaire) :\n"
            f"{self.slots.as_text()}"
        )
        msgs = [SystemMessage(content=system)] + self.summary_mem.context_messages()[1:]
        msgs.append(HumanMessage(content=user_text))

        # réponse
        resp = self.llm.invoke(msgs)
        answer = resp.content.strip()

        # maj mémoire
        self.summary_mem.add_user(user_text)
        self.summary_mem.add_ai(answer)
        self.summary_mem.maybe_summarize()

        return answer

def main():
    agent = Agent()
    print("=== Labo 6 : Persistance JSON (gemma3 @ Ollama) ===")
    print(f"(Fichier mémoire : {MEMORY_PATH})")
    # Affiche ce qui a été rechargé
    reloaded = {
        "summary": agent.summary_mem.summary,
        "slots": agent.slots.to_dict(),
        "buffer_len": len(agent.summary_mem.buffer),
    }
    print("Mémoire rechargée au démarrage :")
    print(json.dumps(reloaded, ensure_ascii=False, indent=2))

    print("\nCommandes utiles :")
    print("- Tape du texte libre (ex: \"Je m'appelle André.\")")
    print("- save   -> écrit la mémoire dans memory.json et l’affiche")
    print("- reset  -> efface la mémoire + supprime le fichier JSON")
    print("- exit   -> quitte")

    try:
        while True:
            user = input("\nVous : ").strip()
            if user.lower() == "exit":
                break
            if user.lower() == "save":
                payload = agent.save()
                print("\n[Sauvegardé] memory.json =")
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                continue
            if user.lower() == "reset":
                agent.reset()
                print("[OK] Mémoire effacée et fichier supprimé.")
                continue

            ans = agent.respond(user)
            print(f"Assistant : {ans}")

    except (KeyboardInterrupt, EOFError):
        print("\nAu revoir !")

if __name__ == "__main__":
    main()
