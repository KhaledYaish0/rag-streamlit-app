import os
import json
import argparse
from typing import List, Tuple

import torch
import faiss
from tqdm import tqdm
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import PyPDF2

# ----------------- Configuration -----------------
PDF_PATH = "2024-wttc-introduction-to-ai.pdf"
STORE_DIR = "rag_index_final"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
EMBED_MODEL_NAME = "all-mpnet-base-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ----------------- Utils -----------------
def read_pdf(path: str) -> List[str]:
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = [page.extract_text() for page in reader.pages]
    print(f"Read {len(pages)} pages")
    return pages

def chunk_text(pages: List[str], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    chunks = []
    for i, page in enumerate(pages):
        words = page.split()
        j = 0
        while j < len(words):
            chunk = words[j:j + chunk_size]
            chunks.append((i, " ".join(chunk)))
            j += (chunk_size - overlap)
    print(f"Created {len(chunks)} chunks")
    return chunks

def clean_chunks(chunks):
    filtered = [(i, ch) for i, ch in chunks if len(ch.split()) >= 40 and not ch.isspace()]
    print(f"Filtered chunks: {len(chunks)} -> {len(filtered)}")
    return filtered

def encode_chunks(embedder, chunks):
    texts = [ch for _, ch in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def save_index(index, metadata):
    os.makedirs(STORE_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(STORE_DIR, "index.faiss"))
    with open(os.path.join(STORE_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_index():
    index = faiss.read_index(os.path.join(STORE_DIR, "index.faiss"))
    with open(os.path.join(STORE_DIR, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

# ----------------- RAG Core -----------------
def build_index_flow():
    pages = read_pdf(PDF_PATH)
    chunks = chunk_text(pages)
    chunks = clean_chunks(chunks)

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    embedder.to(torch.device("cpu"))

    vectors = encode_chunks(embedder, chunks)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    save_index(index, {
        "model_name": EMBED_MODEL_NAME,
        "chunks": [
            {"chunk_id": f"p{pid}_c{i}", "text": ch, "page": pid}
            for i, (pid, ch) in enumerate(chunks)
        ]
    })
    print("Index built successfully.")
    return embedder, index, chunks

def search_top_k(embedder, index, meta, question, k=3) -> List[dict]:
    q_emb = embedder.encode([question])
    D, I = index.search(q_emb, k)
    results = []
    for i in I[0]:
        result = meta["chunks"][i]
        result["score"] = float(D[0][list(I[0]).index(i)])
        results.append(result)
    return results

def rerank(cross_encoder, question, passages):
    pairs = [[question, p["text"]] for p in passages]
    scores = cross_encoder.predict(pairs)
    for i in range(len(passages)):
        passages[i]["score"] = float(scores[i])
    return sorted(passages, key=lambda x: x["score"], reverse=True)

def answer_question(ranked):
    context = "\n".join([x["text"] for x in ranked[:3]])
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(context, max_length=250, min_length=30, do_sample=False)[0]["summary_text"]
    return summary

# ----------------- Streamlit UI -----------------
def app():
    st.title("🔍 RAG Document Q&A")
    st.write("Ask a question based on the loaded PDF document.")

    user_question = st.text_input("📝 Your Question")
    if user_question:
        embedder = SentenceTransformer(EMBED_MODEL_NAME)
        embedder.to(torch.device("cpu"))
        index, meta = load_index()

        top_k = search_top_k(embedder, index, meta, user_question, k=6)
        cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
        reranked = rerank(cross_encoder, user_question, top_k)
        final_answer = answer_question(reranked)

        st.subheader("💡 Answer")
        st.write(final_answer)

        st.subheader("📚 Evidence")
        for r in reranked[:4]:
            st.markdown(f"**Page:** p{r['page']} — Score: {r['score']:.4f}\n\n> {r['text'][:300]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Rebuild the FAISS index")
    args = parser.parse_args()

    if args.build:
        build_index_flow()
    else:
        app()