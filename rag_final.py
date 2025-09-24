import os
import json
import torch
import faiss
import tempfile

from typing import List, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import PyPDF2

class RagEngine:
    def __init__(self, pdf_path: str, build_index: bool = True):
        self.pdf_path = pdf_path
        self.embed_model = "all-mpnet-base-v2"
        self.cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        self.embedder = SentenceTransformer(self.embed_model)
        self.cross_encoder = CrossEncoder(self.cross_encoder_model)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.embedder.to(torch.device("cpu"))

        if build_index:
            self._build_index()
        else:
            raise ValueError("Dynamic mode requires index to be built from PDF.")

    def _read_pdf(self) -> List[str]:
        with open(self.pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = [page.extract_text() for page in reader.pages]
        return pages

    def _chunk_text(self, pages: List[str], chunk_size=300, overlap=40) -> List[Tuple[int, str]]:
        chunks = []
        for i, page in enumerate(pages):
            words = page.split()
            j = 0
            while j < len(words):
                chunk = words[j:j + chunk_size]
                chunks.append((i, " ".join(chunk)))
                j += (chunk_size - overlap)
        return chunks

    def _clean_chunks(self, chunks):
        return [(i, ch) for i, ch in chunks if len(ch.split()) >= 40 and not ch.isspace()]

    def _build_index(self):
        pages = self._read_pdf()
        raw_chunks = self._chunk_text(pages)
        self.chunks = self._clean_chunks(raw_chunks)

        texts = [ch for _, ch in self.chunks]
        self.embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        self.metadata = [
            {"chunk_id": f"p{pid}_c{i}", "text": ch, "page": pid}
            for i, (pid, ch) in enumerate(self.chunks)
        ]

    def query(self, question: str, return_evidence: bool = True, k: int = 5):
        q_emb = self.embedder.encode([question])
        D, I = self.index.search(q_emb, k)

        passages = []
        for i in I[0]:
            result = self.metadata[i]
            result["score"] = float(D[0][list(I[0]).index(i)])
            passages.append(result)

        reranked = self._rerank(question, passages)

        context = "\n".join([x["text"] for x in reranked[:3]])
        summary = self.summarizer(context, max_length=250, min_length=30, do_sample=False)[0]["summary_text"]

        if return_evidence:
            return summary, reranked[:4]
        else:
            return summary

    def _rerank(self, question, passages):
        pairs = [[question, p["text"]] for p in passages]
        scores = self.cross_encoder.predict(pairs)
        for i in range(len(passages)):
            passages[i]["score"] = float(scores[i])
        return sorted(passages, key=lambda x: x["score"], reverse=True)
