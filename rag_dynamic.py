import os
import json
import tempfile
import torch
import faiss
from typing import List
from tqdm import tqdm
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import PyPDF2

# CONFIG
CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
EMBED_MODEL_NAME = "all-mpnet-base-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Read PDF
def read_pdf(uploaded_file) -> List[str]:
    reader = PyPDF2.PdfReader(uploaded_file)
    pages = [page.extract_text() for page in reader.pages]
    return pages

# Chunking
def chunk_text(pages: List[str]) -> List[str]:
    chunks = []
    for i, page in enumerate(pages):
        words = page.split()
        j = 0
        while j < len(words):
            chunk = words[j:j + CHUNK_SIZE]
            chunks.append((i, " ".join(chunk)))
            j += (CHUNK_SIZE - CHUNK_OVERLAP)
    return chunks

def clean_chunks(chunks):
    return [(i, ch) for i, ch in chunks if len(ch.split()) >= 40 and not ch.isspace()]

def encode_chunks(embedder, chunks):
    texts = [ch for _, ch in chunks]
    return embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

def search_top_k(embedder, index, meta, question, k=3):
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
    return summarizer(context, max_length=250, min_length=30, do_sample=False)[0]["summary_text"]

# Streamlit UI
st.set_page_config(page_title="RAG PDF QA", layout="centered")
st.title("📄 RAG PDF Question Answering")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading PDF and preparing index..."):
        pages = read_pdf(uploaded_file)
        chunks = clean_chunks(chunk_text(pages))

        embedder = SentenceTransformer(EMBED_MODEL_NAME)
        embedder.to(torch.device("cpu"))

        vectors = encode_chunks(embedder, chunks)
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

        metadata = {
            "model_name": EMBED_MODEL_NAME,
            "chunks": [{"chunk_id": f"p{pid}_c{i}", "text": ch, "page": pid} for i, (pid, ch) in enumerate(chunks)]
        }

        st.success("Index ready. Ask your question!")

        question = st.text_input("❓ Your Question")
        if question:
            top_k = search_top_k(embedder, index, metadata, question, k=6)
            cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
            reranked = rerank(cross_encoder, question, top_k)
            final_answer = answer_question(reranked)

            st.subheader("💡 Answer")
            st.write(final_answer)

            st.subheader("📚 Sources")
            for r in reranked[:3]:
                st.markdown(f"**Page:** {r['page']} — Score: {r['score']:.4f}\n\n> {r['text'][:300]}...")

