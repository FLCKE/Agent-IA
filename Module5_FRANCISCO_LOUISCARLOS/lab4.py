from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM local via Ollama
llm = ChatOllama(model="gemma3:1b", temperature=0.2)
parser = StrOutputParser()

# Étape 1 : Résumer
prompt_resume = ChatPromptTemplate.from_messages([
    ("system", "Tu résumes des textes. Réponds en français, 2 phrases max."),
    ("user", "Résume ce texte :\n\n{input_text}")
])
resume_chain = prompt_resume | llm | parser

# Étape 2 : Traduire le résumé en anglais
prompt_translate = ChatPromptTemplate.from_messages([
    ("system", "Tu traduis fidèlement en anglais, sans ajouter d'informations."),
    ("user", "Traduis en anglais ce résumé :\n\n{summary_fr}")
])
translate_chain = prompt_translate | llm | parser

# Workflow séquentiel : penser → agir → répondre
text = "Les agents IA transforment l’automatisation des entreprises."

summary_fr = resume_chain.invoke({"input_text": text})
summary_en = translate_chain.invoke({"summary_fr": summary_fr})

print("🧩 Étape 1 — Résumé (FR):", summary_fr)
print("🧩 Étape 2 — Traduction (EN):", summary_en)
print("\n✅ Sortie finale:", summary_en)
