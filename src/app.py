# app.py
import streamlit as st
import subprocess

# Après :
# run_id = log_to_langsmith(user_msg, reply, model)
# st.caption(f"🧩 Run logué sur LangSmith : {run_id}")



st.set_page_config(page_title="Chat Ollama", page_icon="🤖")
st.title("Assistant IA (Ollama) 🤖")
st.caption("Modèles locaux via Ollama — ex: mistral, gemma2:2b, llama3.1")

# --- Sidebar : paramètres ---
with st.sidebar:
    model = st.selectbox(
        "Choisir un modèle Ollama :",
        ["mistral", "gemma3", "llama3"],
        index=0  # mistral par défaut
    )
    st.markdown("Exemples : `mistral`, `gemma3`, `llama3`")
    st.divider()

# --- Mémoire de chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user/assistant", "content": "..."}]

def call_ollama(model_name: str, prompt: str) -> str:
    """Appelle Ollama en CLI et retourne la réponse texte ou un message d'erreur."""
    try:
        r = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=120,
        )
        if r.returncode != 0:
            # Erreurs renvoyées par Ollama
            return f"⚠️ Erreur Ollama ({r.returncode}) : {(r.stderr or r.stdout).strip()}"
        return r.stdout.strip() or "(réponse vide)"
    except FileNotFoundError:
        return "⚠️ Ollama introuvable. Installe-le et vérifie que la commande `ollama` est accessible."
    except subprocess.TimeoutExpired:
        return "⏳ Délai dépassé lors de l'appel à Ollama."
    except Exception as e:
        return f"⚠️ Exception: {e}"

# --- Afficher l'historique ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Entrée utilisateur ---
user_msg = st.chat_input("Vous :")
if user_msg:
    # Affiche + stocke le message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Réponse du modèle
    with st.chat_message("assistant"):
        with st.spinner("L'agent réfléchit..."):
            reply = call_ollama(model, user_msg)
        st.markdown(reply)
    prompt="Analyse moi cette reponse de la question precedente et donne moi une note de 1 a 5 en pertinence, exactitude, clarté, cohérence, style/ton. Reponds au format JSON { 'pertinence':X, 'exactitude':X, 'clarte':X, 'coherence':X, 'style_ton':X } ou X est la note correspondante. Justifie chaque note en une phrase courte apres le JSON. Voici la reponse a analyser : " + reply
    reply = call_ollama(model, prompt )
    st.write(reply)

    # Stocke la réponse
    st.session_state.messages.append({"role": "assistant", "content": reply})
