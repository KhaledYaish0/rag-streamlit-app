import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import torch

# ----------------- Dummy fallback -----------------
DUMMY_CHUNKS = [
    {"page": 1, "text": "Artificial Intelligence is the simulation of human intelligence processes by machines."},
    {"page": 3, "text": "RAG systems combine retrieval mechanisms with generation models to answer complex queries."},
    {"page": 5, "text": "Transformers such as BART and GPT have revolutionized natural language understanding and generation."}
]

# ----------------- RAG Logic -----------------
def dummy_answer_question(question):
    context = "\n".join([x["text"] for x in DUMMY_CHUNKS[:3]])
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(context, max_length=250, min_length=30, do_sample=False)[0]["summary_text"]
    return summary

# ----------------- Streamlit UI -----------------
def app():
    st.title("🔍 RAG Demo App (Cloud-Friendly)")
    st.write("This is a demo version of the app using dummy data (no FAISS/index files required).")

    user_question = st.text_input("📝 Your Question")
    if user_question:
        st.info("⚠️ FAISS index not loaded. Showing answer from dummy content instead.")
        final_answer = dummy_answer_question(user_question)

        st.subheader("💡 Answer")
        st.write(final_answer)

        st.subheader("📚 Evidence")
        for r in DUMMY_CHUNKS:
            st.markdown(f"**Page:** p{r['page']}\n\n> {r['text'][:300]}...")

if __name__ == "__main__":
    app()
