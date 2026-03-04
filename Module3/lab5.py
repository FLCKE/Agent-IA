# eval_labo5.py
import csv
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import ollama

PDF_PATH = "document.pdf"

def chunk_text(t: str, chunk_size: int = 500, overlap: int = 100):
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(t):
        chunks.append(t[start:start+chunk_size])
        start += step
    return chunks

# 1) Load PDF
reader = PdfReader(PDF_PATH)
text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
if not text:
    raise ValueError("Aucun texte extrait du PDF (scan => OCR).")

chunks = chunk_text(text)
print("chunks:", len(chunks))

# 2) Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embs = model.encode(chunks, convert_to_numpy=True)

def retrieve(query: str, k: int = 3):
    q_emb = model.encode([query], convert_to_numpy=True)[0]
    q_norm = np.linalg.norm(q_emb) + 1e-12
    c_norms = np.linalg.norm(chunk_embs, axis=1) + 1e-12
    sims = (chunk_embs @ q_emb) / (c_norms * q_norm)
    top_idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i]), chunks[i]) for i in top_idx]

def rag_answer(query: str, k: int = 3):
    sources = retrieve(query, k=k)
    context = "\n\n---\n\n".join(
        [f"[SOURCE {r+1} | chunk={idx} | sim={sim:.3f}]\n{chunk}"
         for r, (idx, sim, chunk) in enumerate(sources)]
    )
    prompt = f"""Réponds uniquement avec le CONTEXTE.
Si l'information n'est pas dans le contexte, dis: "Je ne trouve pas l'information dans le document fourni."

CONTEXTE:
{context}

QUESTION:
{query}

RÉPONSE:
"""
    resp = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )
    return resp["message"]["content"], sources

QUESTIONS = [
    "Quels sont les droits principaux d’un utilisateur selon le document ?",
    "Dans quels cas un utilisateur peut-il demander la suppression de ses données ?",
    "Quel délai est indiqué pour répondre à une demande utilisateur ?",
    "À qui l’utilisateur peut-il adresser une réclamation ?",
    "Le document mentionne-t-il des conditions ou limites à ces droits ?",
]

rows = []
for q in QUESTIONS:
    ans, srcs = rag_answer(q, k=3)
    src_str = " | ".join([f"chunk={i} sim={s:.3f}" for i, s, _ in srcs])
    rows.append([q, ans.replace("\n", " "), src_str])

with open("evaluation_labo5_sortie.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Question", "Reponse_agent", "Sources(top3)"])
    writer.writerows(rows)

print("✅ Fichier créé : evaluation_labo5_sortie.csv")
