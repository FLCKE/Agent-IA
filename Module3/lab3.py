import os
from PyPDF2 import PdfReader

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------- 1) Charger le PDF et extraire le texte --------
PDF_PATH = "document.pdf"  # <-- mets ici ton PDF
reader = PdfReader(PDF_PATH)

pages_text = []
for i, page in enumerate(reader.pages):
    t = page.extract_text() or ""
    pages_text.append(t)

text = "\n".join(pages_text).strip()
if not text:
    raise ValueError("Aucun texte extrait du PDF (peut être un PDF scanné).")


# -------- 2) Chunking --------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],  # découpe progressive
)
chunks = splitter.split_text(text)

print(f"✅ Nombre de chunks générés : {len(chunks)}")


# -------- 3) Embeddings (local) --------
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    # Chroma attend list[list[float]]
    return model.encode(texts, convert_to_numpy=True).tolist()


# -------- 4) Indexation dans Chroma (persistant) --------
PERSIST_DIR = "./chroma_rgpd_db"   # dossier qui contiendra la base
COLLECTION_NAME = "docs_chunks"

client = chromadb.PersistentClient(
    path=PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Ids + métadonnées (optionnel mais utile)
ids = [f"chunk_{i}" for i in range(len(chunks))]
metadatas = [{"source": PDF_PATH, "chunk_index": i} for i in range(len(chunks))]

embeddings = embed(chunks)

collection.add(
    ids=ids,
    documents=chunks,
    metadatas=metadatas,
    embeddings=embeddings
)

print(f"✅ Chunks indexés dans Chroma : {collection.count()}")
print(f"📁 Base persistée dans : {os.path.abspath(PERSIST_DIR)}")


# -------- 5) Petite vérif (optionnelle) : recherche --------
query = "durée du contrat"
q_emb = embed([query])[0]
results = collection.query(query_embeddings=[q_emb], n_results=3)

print("\n🔎 Exemple de recherche :")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"- chunk {meta['chunk_index']} | extrait: {doc[:120].replace('\\n',' ')}...")
