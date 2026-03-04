import ollama

MODEL = "gemma3:1b"

def call_ollama(prompt: str, model: str = MODEL) -> str:
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp["message"]["content"].strip()

def tool_meteo(ville: str) -> str:
    if ville.lower() == "paris":
        return "Il pleut à Paris ☔️"
    return f"Météo indisponible pour {ville}"

def react_agent(question: str) -> str:
    print("🧠 Raisonnement:", f"Je vais réfléchir à la question : {question}")

    q_lower = question.lower()
    action = "appeler outil météo" if ("météo" in q_lower or "meteo" in q_lower) else "répondre directement"
    print("⚙️ Action:", action)

    if action == "appeler outil météo":
        ville = "Paris" if "paris" in q_lower else "inconnue"
        observation = tool_meteo(ville)
    else:
        observation = call_ollama(f"Réponds brièvement à cette question:\n{question}")
    print("👀 Observation:", observation)

    reflection = call_ollama(
        f"Question: {question}\nAction: {action}\nObservation: {observation}\nConclusion finale (1 phrase):"
    )
    print("🔍 Réflexion:", reflection)

    return reflection

react_agent("Quelle est la météo à Paris ?")
