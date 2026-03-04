"""
Utilitaires Ollama pour le Module 8.

Ce module centralise les appels HTTP vers Ollama afin d'éviter
de dupliquer le code dans chaque labo.
"""

from __future__ import annotations

import json
import os
from urllib import request, error


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _list_models() -> list[str]:
    req = request.Request(url=f"{OLLAMA_URL}/api/tags", method="GET")
    with request.urlopen(req, timeout=15) as response:
        body = json.loads(response.read().decode("utf-8"))
    return [m.get("name", "") for m in body.get("models", [])]


def ollama_generate(prompt: str, model: str = CHAT_MODEL, temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    req = request.Request(
        url=f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body.get("response", "").strip()


def ollama_embedding(text: str, model: str = EMBED_MODEL) -> list[float]:
    payload = {"model": model, "prompt": text}

    # Compatibilité versions Ollama:
    # - anciennes versions: /api/embeddings (champ embedding)
    # - nouvelles versions: /api/embed (champ embeddings)
    endpoints = [
        ("/api/embeddings", "embedding"),
        ("/api/embed", "embeddings"),
    ]

    last_error: Exception | None = None
    for endpoint, key in endpoints:
        try:
            req = request.Request(
                url=f"{OLLAMA_URL}{endpoint}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))

            if key == "embedding" and key in body:
                return body[key]
            if key == "embeddings" and key in body and body[key]:
                return body[key][0]
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Embedding indisponible via Ollama ({last_error})")


def check_ollama_ready() -> tuple[bool, str]:
    try:
        models = _list_models()
        if CHAT_MODEL not in models:
            return False, f"Modèle chat manquant: {CHAT_MODEL}"
        if EMBED_MODEL not in models:
            return False, f"Modèle embedding manquant: {EMBED_MODEL}"
        return True, "Ollama prêt"
    except error.URLError:
        return False, f"Impossible de joindre Ollama sur {OLLAMA_URL}"
    except Exception as exc:
        return False, f"Erreur de vérification Ollama: {exc}"


def check_chat_ready() -> tuple[bool, str]:
    try:
        models = _list_models()
        if CHAT_MODEL not in models:
            return False, f"Modèle chat manquant: {CHAT_MODEL}"
        return True, "Ollama prêt (chat)"
    except error.URLError:
        return False, f"Impossible de joindre Ollama sur {OLLAMA_URL}"
    except Exception as exc:
        return False, f"Erreur de vérification Ollama: {exc}"


def check_embedding_ready() -> tuple[bool, str]:
    try:
        models = _list_models()
        if EMBED_MODEL not in models:
            return False, f"Modèle embedding manquant: {EMBED_MODEL}"
        return True, "Ollama prêt (embedding)"
    except error.URLError:
        return False, f"Impossible de joindre Ollama sur {OLLAMA_URL}"
    except Exception as exc:
        return False, f"Erreur de vérification Ollama: {exc}"
