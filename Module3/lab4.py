import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import ollama


# -------------------------
# 1) Charger PDF -> texte
# -------------------------
PDF_PATH = "document.pdf"

reader = PdfReader(PDF_PATH)
text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()

if not text:
    raise ValueError("Aucun texte extrait du PDF (si PDF scanné => OCR nécessaire).")


# -------------------------
# 2) Chunking (simple)
# -------------------------
def chunk_text(t: str, chunk_size: int = 500, overlap: int = 100):
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(t):
        end = start + chunk_size
        chunks.append(t[start:end])
        start += step
    return chunks

chunks = chunk_text(text, chunk_size=500, overlap=100)
print(f"✅ Nombre de chunks indexés : {len(chunks)}")


# -------------------------
# 3) Embeddings
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embs = model.encode(chunks, convert_to_numpy=True)


# -------------------------
# 4) Retrieval top-k (cosine similarity)
# -------------------------
def retrieve(query: str, k: int = 3):
    q_emb = model.encode([query], convert_to_numpy=True)[0]

    q_norm = np.linalg.norm(q_emb) + 1e-12
    c_norms = np.linalg.norm(chunk_embs, axis=1) + 1e-12
    sims = (chunk_embs @ q_emb) / (c_norms * q_norm)

    top_idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i]), chunks[i]) for i in top_idx]


# -------------------------
# 5) Génération (Ollama Gemma)
# -------------------------
def rag_answer(query: str, k: int = 3):
    sources = retrieve(query, k=k)

    context = "\n\n---\n\n".join(
        [f"[SOURCE {rank+1} | chunk={idx} | sim={sim:.3f}]\n{chunk}"
         for rank, (idx, sim, chunk) in enumerate(sources)]
    )

    prompt = f"""Tu es un assistant RAG.
Tu dois répondre UNIQUEMENT à partir du CONTEXTE.
Si l'info n'est pas dans le contexte, réponds : "Je ne trouve pas l'information dans le document fourni."

CONTEXTE:
{context}

QUESTION:
{query}

RÉPONSE (en français, claire et structurée):
"""

    resp = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )
    answer = resp["message"]["content"]
    return answer, sources


if __name__ == "__main__":
    query = "Quels sont les droits d’un utilisateur selon le RGPD ?"
    answer, sources = rag_answer(query, k=3)

    print("\n================= RÉPONSE =================")
    print(answer)

    print("\n================= SOURCES =================")
    for rank, (idx, sim, chunk) in enumerate(sources, 1):
        excerpt = chunk[:220].replace("\n", " ")
        print(f"- Source {rank} | chunk_index={idx} | sim={sim:.3f} | extrait=\"{excerpt}...\"")
