import subprocess
import json
from typing import List, Dict

MODEL = "gemma3:1b"

def ollama_chat(prompt: str, model: str = MODEL) -> str:
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    if result.returncode != 0:
        return f"Erreur: {result.stderr.strip()}"
    return result.stdout.strip()

def get_agent_opinion(agent_role: str, question: str) -> Dict:
    """
    Simule une opinion d'agent avec un score de confiance.
    """
    prompt = f"""
    Tu es un agent avec le rôle : {agent_role}. 
    Réponds brièvement par OUI ou NON à la question suivante : {question}.
    Explique ton choix en une phrase courte.
    Donne aussi ton score de confiance (entre 0 et 100).
    
    Réponds EXCLUSIVEMENT sous le format JSON suivant :
    {{"decision": "OUI/NON", "explanation": "ta raison", "score": 85}}
    """
    response = ollama_chat(prompt)
    try:
        # On essaie de parser le JSON (en gérant les backticks éventuels)
        clean_response = response.strip("`").replace("json\n", "").strip()
        return json.loads(clean_response)
    except:
        # Fallback simple si le modèle ne sort pas du JSON propre
        return {"decision": "OUI", "explanation": "Pas pu parser le JSON", "score": 50}

def consensus_majority(responses: List[str]) -> str:
    """Vote majoritaire simple sur la décision."""
    if not responses: return "N/A"
    return max(set(responses), key=responses.count)

def consensus_arbitrator(question: str, responses: List[Dict]) -> str:
    """Un agent juge choisit la meilleure réponse."""
    prompt = f"""
    En tant que Juge Arbitre, analyse les réponses de ces 3 agents à la question : '{question}'.
    Réponses : {json.dumps(responses)}
    
    Choisis la réponse la plus convaincante et justifie ton choix.
    """
    return ollama_chat(prompt)

def consensus_weighted(responses: List[Dict]) -> Dict:
    """Choisit la réponse avec le score de confiance le plus élevé."""
    return sorted(responses, key=lambda r: r['score'], reverse=True)[0]

if __name__ == "__main__":
    question = "L'intelligence artificielle peut-elle un jour avoir une conscience ?"
    print(f"--- Lab 5 : Consensus sur la question : {question} ---\n")

    roles = ["Optimiste Technologique", "Philosophe Sceptique", "Ingénieur Pragmatique"]
    agent_responses = []

    for role in roles:
        print(f"Agent '{role}' en train de réfléchir...")
        res = get_agent_opinion(role, question)
        agent_responses.append(res)
        print(f"-> {role} a dit : {res.get('decision')} (Score: {res.get('score')})\n")

    # 1. Vote Majoritaire
    decisions = [r['decision'] for r in agent_responses]
    majority = consensus_majority(decisions)
    print(f"1. Résultat (Vote Majoritaire) : {majority}")

    # 2. Score de Confiance (Pondération)
    best_weighted = consensus_weighted(agent_responses)
    print(f"2. Résultat (Score de Confiance) : {best_weighted['decision']} (Score: {best_weighted['score']})")

    # 3. Agent Arbitre
    print("\n3. Appel de l'Agent Arbitre pour délibération...")
    arbitration = consensus_arbitrator(question, agent_responses)
    print(f"\n--- Délibération du Juge ---\n{arbitration}")
