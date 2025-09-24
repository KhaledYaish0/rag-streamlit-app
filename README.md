# 🔍 RAG Streamlit App

A simple and interactive Retrieval-Augmented Generation (RAG) app powered by FAISS, Sentence Transformers, and HuggingFace models – with a Streamlit interface for querying local documents.

---

## 🚀 Features

- 🔎 Semantic search over your own documents using FAISS
- 🧠 Uses SentenceTransformer for dense retrieval
- 📰 Summarization using BART (`facebook/bart-large-cnn`)
- 🖥️ Clean interface with **Streamlit**
- 📂 Index management with metadata JSON and FAISS files

---

## 📁 Project Structure

```
.
├── app_streamlit.py       # Main Streamlit UI
├── rag_final.py           # Final RAG logic (retrieval + generation)
├── rag_dynamic.py         # Flexible version of RAG
├── requirements.txt       # List of Python dependencies
├── meta.json              # Metadata for indexed chunks
├── index.faiss            # FAISS vector index
└── .gitignore             # Files to be ignored by Git
```

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/KhaledYaish0/rag-streamlit-app.git
cd rag-streamlit-app
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install requirements**
```bash
pip install -r requirements.txt
```

---

## 🧠 How it Works

1. Your document is chunked and vectorized using Sentence Transformers.
2. FAISS indexes these chunks for efficient similarity search.
3. When you enter a question, it retrieves top-k chunks and uses BART to summarize the answer.
4. Everything runs locally – no need to call OpenAI API.

---

## 💻 Run the App

```bash
streamlit run app_streamlit.py
```

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## Author

 by [Khaled Yaish](https://github.com/KhaledYaish0)
