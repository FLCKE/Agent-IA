import os
from pathlib import Path

# --- Config runtime ---
OLLAMA_LLM = os.getenv("OLLAMA_LLM", "gemma3")  # ton modèle local (déjà installé)
OLLAMA_EMBED = os.getenv("OLLAMA_EMBED", "nomic-embed-text")  # modèle d'embeddings Ollama
PERSIST_DIR = "./memo_db"  # persistance disque entre sessions

# --- LangChain / Chroma (versions community) ---
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

# 1) Initialiser embeddings + Chroma (persistant)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBED)
Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
store = Chroma(
    collection_name="memo",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

# 2) Initialiser le LLM local (gemma3)
chat = ChatOllama(model=OLLAMA_LLM)
def extract_after_prefix(msg: str, prefix: str) -> str:
    low = msg.lower()
    i = low.find(prefix)
    if i == -1:
        return ""
    # position après le préfixe
    j = i + len(prefix)
    # saute espaces et ponctuation (ex: " : ")
    while j < len(msg) and msg[j] in " .,:;!?\n\t\"'":
        j += 1
    return msg[j:].strip()


def remember(text: str) -> str:
    """
    Ajoute un souvenir au vector store et persiste sur disque.
    """
    if not text.strip():
        return "Rien à mémoriser."
    store.add_texts([text.strip()], metadatas=[{"type": "memory"}])
    # store.persist()
    return "C'est noté, je m'en souviendrai."

def recall(query: str, k: int = 3) -> str:
    """
    Recherche sémantique dans la mémoire.
    """
    docs = store.similarity_search(query, k=k)
    if not docs:
        return "Je n'ai rien trouvé dans ma mémoire."
    # Retour simple : listes des contenus
    lines = []
    for i, d in enumerate(docs, 1):
        lines.append(f"- {i}. {d.page_content}")
    return "Voici ce que j'ai retrouvé :\n" + "\n".join(lines)

def answer_with_mem(context_query: str, user_msg: str) -> str:
    """
    (Optionnel) Utilise les souvenirs pertinents comme contexte pour répondre avec gemma3.
    """
    docs = store.similarity_search(context_query, k=3)
    context = "\n".join(d.page_content for d in docs) if docs else "Aucun souvenir pertinent."
    messages = [
        SystemMessage(content=(
            "Tu es un assistant concis. Utilise le CONTEXTE_MEMOIRE s'il est pertinent. "
            "Si la mémoire ne contient rien d'utile, réponds normalement."
        )),
        HumanMessage(content=f"CONTEXTE_MEMOIRE:\n{context}\n\nQuestion:\n{user_msg}")
    ]
    resp = chat(messages)
    return resp.content

def handle(msg: str) -> str:
    low = msg.lower().strip()

    # Ajout de souvenirs
    if low.startswith("souviens-toi de"):
        fact = extract_after_prefix(msg, "souviens-toi de")
        if not fact:
            return "Je n'ai rien à mémoriser (format: « Souviens-toi de : <fait> »)."
        return remember(fact)

    # Rappel / interrogation de la mémoire
    if low.startswith("rappelle-moi"):
        ask = extract_after_prefix(msg, "rappelle-moi")
        if not ask:
            return "Dis-moi ce que je dois rappeler (format: « Rappelle-moi : <question> »)."
        return recall(ask, k=3)

    # Démo : question libre appuyée par la mémoire (optionnel)
    if low.startswith("qu'aime") or low.startswith("que sait-tu") or "mémoire" in low:
        return answer_with_mem(msg, msg)

    return ("Commandes disponibles :\n"
            "- \"Souviens-toi de : <fait>\"\n"
            "- \"Rappelle-moi <question>\"\n"
            "- Ou pose une question libre, j'essaierai d'utiliser ma mémoire.")

if __name__ == "__main__":
    print("Mémoire long terme avec Ollama (gemma3 + Chroma).")
    print("Exemples:\n - Souviens-toi de : André aime les agents d’IA.\n - Rappelle-moi : Qu’aime André ?\n")
    try:
        while True:
            msg = input("Vous : ")
            print(handle(msg))
    except (KeyboardInterrupt, EOFError):
        print("\nAu revoir !")
