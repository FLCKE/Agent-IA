import os
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatOllama

# Configuration Ollama
llm = ChatOllama(
    model="gemma3:1b",
    base_url="http://localhost:11434"
)

# 1. Définition des Agents
researcher = Agent(
    role="Chercheur",
    goal="Collecter des informations pertinentes sur les tendances de l'IA en 2025.",
    backstory="Expert en veille technologique spécialisé dans l'analyse prospective.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

writer = Agent(
    role="Rédacteur",
    goal="Synthétiser les informations collectées en un article court.",
    backstory="Rédacteur scientifique capable de vulgariser des concepts complexes.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

reviewer = Agent(
    role="Relecteur",
    goal="Vérifier la cohérence et la qualité du texte final.",
    backstory="Éditeur exigeant avec un œil critique sur le style.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# 2. Définition des Tâches
task1 = Task(
    description="Rechercher les 3 innovations majeures prévues en IA pour 2025.",
    expected_output="Une liste de 3 innovations avec descriptions.",
    agent=researcher
)

task2 = Task(
    description="Rédiger un article de blog de 150 mots sur ces innovations.",
    expected_output="Un article de blog structuré.",
    agent=writer
)

task3 = Task(
    description="Relire l'article pour validation finale.",
    expected_output="L'article final corrigé.",
    agent=reviewer
)

# 3. Création de l'Équipe
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[task1, task2, task3],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("=== Lancement de l'équipe CrewAI ===\n")
    result = crew.kickoff()
    print("\n--- RÉSULTAT FINAL ---\n")
    print(result)
