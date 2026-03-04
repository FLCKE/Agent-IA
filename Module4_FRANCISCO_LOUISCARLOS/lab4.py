import json
import requests

MODEL = "gemma3:1b"

SCHEMA_EXAMPLE = {
    "task": "send_email",
    "recipient": "marie@exemple.com",
    "message": "Rappel de réunion",
    "time": "2025-10-10T09:00:00"
}

def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es un générateur de JSON strict.\n"
                    "Règles:\n"
                    "- Réponds UNIQUEMENT avec un JSON valide.\n"
                    "- Aucun texte avant/après.\n"
                    "- Utilise exactement les clés: task, recipient, message, time.\n"
                    "- time doit être au format ISO 8601: YYYY-MM-DDTHH:MM:SS\n"
                    f"- Exemple de format attendu: {json.dumps(SCHEMA_EXAMPLE, ensure_ascii=False)}"
                )
            },
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.1}
    }

    r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["message"]["content"]

def extract_json(text: str) -> str:
    """
    Fallback si le modèle ajoute du texte.
    On récupère le premier {...} trouvé.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start:end + 1]

def validate_json(response_text: str) -> dict:
    # 1) tente direct
    try:
        data = json.loads(response_text)
        return data
    except json.JSONDecodeError:
        # 2) fallback extraction
        cleaned = extract_json(response_text)
        if not cleaned:
            raise json.JSONDecodeError("Aucun JSON détecté", response_text, 0)
        return json.loads(cleaned)

def check_schema(data: dict) -> None:
    required_keys = ["task", "recipient", "message", "time"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Clés manquantes: {missing}")

    if not isinstance(data["task"], str):
        raise ValueError("task doit être une string")
    if not isinstance(data["recipient"], str) or "@" not in data["recipient"]:
        raise ValueError("recipient doit être un email valide")
    if not isinstance(data["message"], str):
        raise ValueError("message doit être une string")
    if not isinstance(data["time"], str) or "T" not in data["time"]:
        raise ValueError("time doit être une string ISO 8601 (ex: 2025-10-10T09:00:00)")

if __name__ == "__main__":
    user_prompt = (
        "Prépare une tâche pour envoyer un email à marie@exemple.com "
        "pour lui rappeler la réunion du 10 octobre 2025 à 09:00."
    )

    response = call_ollama(user_prompt)
    print("=== Réponse brute du LLM ===")
    print(response)

    print("\n=== Validation JSON ===")
    try:
        data = validate_json(response)
        check_schema(data)
        print("✅ JSON valide et schéma OK")
        print("JSON final:", json.dumps(data, ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print("⚠️ JSON invalide, correction nécessaire.")
    except ValueError as e:
        print(f"⚠️ JSON valide mais schéma incorrect: {e}")
