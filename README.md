# 🧠 RAG Streamlit App

This is a Retrieval-Augmented Generation (RAG) app built with **Streamlit**, allowing you to upload a PDF and ask questions about its content using semantic search and summarization.

## 🚀 Features
- 📄 Upload any PDF document.
- 🔍 Ask natural language questions about the content.
- 🤖 Uses Sentence Transformers + CrossEncoder reranking.
- 📚 Generates answers with BART summarizer.
- ⚡ Built-in FAISS for fast similarity search.
- 🌐 Fully deployable on [Streamlit Cloud](https://streamlit.io/cloud).

## 📂 Project Structure
```
├── app_streamlit.py       # Main Streamlit frontend app
├── rag_final.py           # RAG backend logic (embedding, indexing, reranking, answering)
├── requirements.txt       # Dependencies
├── README.md              # Project overview
├── .gitignore
├── index.faiss            # Vector index (auto-generated)
├── meta.json              # Chunk metadata (auto-generated)
```

## 📦 Requirements
- Python 3.8+
- PyTorch
- Transformers
- Sentence-Transformers
- FAISS
- Streamlit

## 🌍 Live Demo
Check out the deployed app here:
👉 [rag-app-streamlit-khaledyaish](https://rag-app-stremlit-khaledyaish.streamlit.app/)

## 🛠️ How to Run Locally
```bash
git clone https://github.com/KhaledYaish0/rag-streamlit-app.git
cd rag-streamlit-app
pip install -r requirements.txt
streamlit run app_streamlit.py
```
