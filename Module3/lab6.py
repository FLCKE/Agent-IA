import re
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import ollama

PDF_PATH = "document.pdf"
CHUNK_SIZE = 500
OVERLAP = 100

# -------------------------
# 1) Utils: chunking / tokenization
# -------------------------
def chunk_text(t: str, chunk_size: int = 500, overlap: int = 100):
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(t):
        chunks.append(t[start:start+chunk_size])
        start += step
    return chunks

def tokenize(text: str):
    # tokenisation simple (FR): mots en minuscules
    return re.findall(r"\w+", text.lower())

def ollama_generate(prompt: str, temperature: float = 0.2):
    resp = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature}
    )
    return resp["message"]["content"].strip()

# -------------------------
# 2) Load PDF -> chunks -> indexes
# -------------------------
reader = PdfReader(PDF_PATH)
text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
if not text:
    raise ValueError("Aucun texte extrait du PDF (scan => OCR).")

chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
print(f"✅ Chunks: {len(chunks)}")

# Embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embs = embed_model.encode(chunks, convert_to_numpy=True)

# BM25
tokenized_chunks = [tokenize(c) for c in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# -------------------------
# 3) Retrieval methods
# -------------------------
def cosine_sims(q_emb):
    q_norm = np.linalg.norm(q_emb) + 1e-12
    c_norms = np.linalg.norm(chunk_embs, axis=1) + 1e-12
    return (chunk_embs @ q_emb) / (c_norms * q_norm)

def retrieve_vector(query: str, k: int):
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    sims = cosine_sims(q_emb)
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i]), chunks[i]) for i in idx]

def retrieve_bm25(query: str, k: int):
    q_tok = tokenize(query)
    scores = bm25.get_scores(q_tok)
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i]), chunks[i]) for i in idx]

def hybrid_retrieve(query: str, k: int = 10, alpha: float = 0.6):
    """
    alpha = poids vectoriel, (1-alpha) = poids lexical
    On normalise chaque score sur [0,1] avant combinaison.
    """
    # Vector
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    v = cosine_sims(q_emb)

    # BM25
    b = bm25.get_scores(tokenize(query))

    # Normalisation min-max
    def norm(x):
        x = np.array(x, dtype=float)
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    v_n = norm(v)
    b_n = norm(b)

    combo = alpha * v_n + (1 - alpha) * b_n
    idx = np.argsort(-combo)[:k]
    return [(int(i), float(combo[i]), chunks[i]) for i in idx]

# -------------------------
# 4) Reranking (re-score top-10 -> keep top-3)
# -------------------------
def rerank_topk(candidates, query: str, keep: int = 3):
    """
    candidates: list[(idx, score, chunk)] size=k
    reranking: recompute cosine similarity between query and candidate chunk only
    """
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    q_norm = np.linalg.norm(q_emb) + 1e-12

    rescored = []
    for idx, _, chunk in candidates:
        c_emb = embed_model.encode([chunk], convert_to_numpy=True)[0]
        sim = float((c_emb @ q_emb) / ((np.linalg.norm(c_emb)+1e-12) * q_norm))
        rescored.append((idx, sim, chunk))

    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:keep]

# -------------------------
# 5) Query rewriting (LLM)
# -------------------------
def rewrite_query(query: str):
    prompt = f"""Reformule la requête suivante pour une recherche documentaire.
Objectif: extraire les mots-clés utiles, enlever le flou, rester court.
Requête: {query}
Requête reformulée:"""
    out = ollama_generate(prompt, temperature=0.0)
    return out.split("\n")[0].strip().strip('"')

# -------------------------
# 6) Answer generation with sources
# -------------------------
def answer_from_sources(query: str, sources):
    context = "\n\n---\n\n".join(
        [f"[SOURCE {i+1} | chunk={idx} | score={score:.3f}]\n{chunk}"
         for i, (idx, score, chunk) in enumerate(sources)]
    )
    prompt = f"""Tu es un assistant RAG.
Réponds uniquement avec le CONTEXTE.
Si l'info n'est pas dans le contexte, dis: "Je ne trouve pas l'information dans le document fourni."

CONTEXTE:
{context}

QUESTION:
{query}

RÉPONSE (français, claire, structurée):
"""
    return ollama_generate(prompt, temperature=0.2)

# -------------------------
# 7) Pipelines: Basique vs Avancé
# -------------------------
def rag_basique(query: str):
    # top-3 vector direct
    src = retrieve_vector(query, k=3)
    ans = answer_from_sources(query, src)
    return ans, src

def rag_avance(query: str):
    # 1) rewrite
    rq = rewrite_query(query)

    # 2) hybrid retrieve top-10
    cand = hybrid_retrieve(rq, k=10, alpha=0.6)

    # 3) rerank -> keep top-3
    src = rerank_topk(cand, rq, keep=3)

    # answer with original question (pas la reformulée)
    ans = answer_from_sources(query, src)
    return rq, ans, src

# -------------------------
# 8) Evaluate: "Pertinence" quick (manual scoring)
# -------------------------
QUESTIONS = [
    "Quels sont les droits principaux d’un utilisateur selon le document ?",
    "Quel délai est indiqué pour répondre à une demande utilisateur ?",
    "À qui l’utilisateur peut-il adresser une réclamation ?",
    "Quelles conditions/limites sont mentionnées concernant ces droits ?",
    "Dans quels cas les données peuvent-elles être effacées ?",
]

print("\n================= COMPARATIF =================")
for q in QUESTIONS:
    ans_b, src_b = rag_basique(q)
    rq, ans_a, src_a = rag_avance(q)

    print("\n----------------------------------------------")
    print("QUESTION :", q)

    print("\n[BASIQUE] Réponse :", ans_b)
    print("[BASIQUE] Sources :", " | ".join([f"chunk={i}({s:.3f})" for i,s,_ in src_b]))

    print("\n[AVANCÉ] Query rewrite :", rq)
    print("[AVANCÉ] Réponse :", ans_a)
    print("[AVANCÉ] Sources :", " | ".join([f"chunk={i}({s:.3f})" for i,s,_ in src_a]))
