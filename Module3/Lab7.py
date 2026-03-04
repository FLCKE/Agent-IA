import re
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import ollama

PDF_PATH = "document.pdf"
CHUNK_SIZE = 500
OVERLAP = 100

# -------------------------
# 1) Chunking + embeddings + retrieval
# -------------------------
def chunk_text(t: str, chunk_size: int = 500, overlap: int = 100):
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(t):
        chunks.append(t[start:start+chunk_size])
        start += step
    return chunks

def cosine_topk(query: str, chunks, chunk_embs, model, k: int = 3):
    q_emb = model.encode([query], convert_to_numpy=True)[0]
    q_norm = np.linalg.norm(q_emb) + 1e-12
    c_norms = np.linalg.norm(chunk_embs, axis=1) + 1e-12
    sims = (chunk_embs @ q_emb) / (c_norms * q_norm)
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i]), chunks[i]) for i in idx]

def ollama_chat(prompt: str, temperature: float = 0.2):
    resp = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return resp["message"]["content"].strip()

# Load PDF
reader = PdfReader(PDF_PATH)
text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
if not text:
    raise ValueError("Aucun texte extrait du PDF (scan => OCR).")

chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
print(f"✅ Chunks: {len(chunks)}")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embs = embed_model.encode(chunks, convert_to_numpy=True)

# -------------------------
# 2) RAG avec ancrage + citations
# -------------------------
def rag_answer_with_citations(query: str, k: int = 3):
    sources = cosine_topk(query, chunks, chunk_embs, embed_model, k=k)

    # On fournit les extraits, et on demande au modèle de citer [S1], [S2], [S3]
    context = "\n\n---\n\n".join(
        [f"[S{i+1} | chunk={idx} | sim={sim:.3f}]\n{chunk}"
         for i, (idx, sim, chunk) in enumerate(sources)]
    )

    prompt = f"""Réponds UNIQUEMENT à partir des documents fournis.
Règles:
- Si l'information n'est pas explicitement dans les sources, réponds: "Je ne trouve pas l'information dans le document fourni."
- Dans ta réponse, ajoute des citations sous forme [S1], [S2], [S3] après chaque affirmation importante.
- À la fin, inclus une section "EXTRAITS UTILISÉS" où tu recopies 1 à 3 phrases exactes des sources (max 2 lignes par source).

SOURCES:
{context}

QUESTION:
{query}

RÉPONSE (français, claire et structurée):
"""
    answer = ollama_chat(prompt, temperature=0.2)
    return answer, sources

# -------------------------
# 3) Vérification croisée (faithfulness check)
# -------------------------
def verify_faithfulness(query: str, answer: str, sources):
    src_text = "\n\n---\n\n".join(
        [f"[S{i+1}]\n{chunk}" for i, (_, _, chunk) in enumerate(sources)]
    )
    verifier_prompt = f"""Tu es un vérificateur strict.
Ta tâche: dire si la RÉPONSE est fidèle aux SOURCES (aucune info inventée).
Réponds en JSON STRICT (sans texte autour) avec les champs:
- faithful: true/false
- issues: liste courte des problèmes (si false)
- suggested_fix: une version corrigée de la réponse (uniquement basée sur les sources) si false, sinon chaîne vide.

QUESTION:
{query}

SOURCES:
{src_text}

RÉPONSE À VÉRIFIER:
{answer}
"""
    verdict = ollama_chat(verifier_prompt, temperature=0.0)
    return verdict

# -------------------------
# 4) Correction automatique si hallucination détectée
# -------------------------
def parse_json_loose(s: str):
    # parse "loose" très simple (sans dépendance) : extrait le bloc { ... }
    m = re.search(r"\{.*\}", s, flags=re.S)
    return m.group(0) if m else s

# -------------------------
# 5) Test sur quelques questions (ex: celles du Labo 6/5)
# -------------------------
QUESTIONS = [
    "Quels sont les droits principaux d’un utilisateur selon le document ?",
    "Quel délai est indiqué pour répondre à une demande utilisateur ?",
    "À qui l’utilisateur peut-il adresser une réclamation ?",
]

results = []
for q in QUESTIONS:
    print("\n==========================================")
    print("QUESTION:", q)

    answer, sources = rag_answer_with_citations(q, k=3)
    print("\n--- RÉPONSE ---")
    print(answer)

    verdict = verify_faithfulness(q, answer, sources)
    verdict_json = parse_json_loose(verdict)

    print("\n--- VÉRIFICATION (JSON) ---")
    print(verdict_json)

    results.append((q, answer, verdict_json, sources))

print("\n✅ Terminé.")
