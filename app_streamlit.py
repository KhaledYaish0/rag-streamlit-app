import os
import tempfile
import streamlit as st
from rag_final import RagEngine

st.set_page_config(page_title="RAG PDF Question Answering", page_icon="🔍")

st.write("Upload a PDF and ask any question about its content.")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

# Initialize RagEngine (will be created only when PDF is uploaded)
rag = None
if uploaded_file:
    # Toast to show PDF is uploaded
    st.toast("PDF uploaded!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success(f"PDF uploaded: {uploaded_file.name}")
    print(f"Uploaded file saved at: {tmp_path}")

    # Build the index dynamically from the uploaded PDF
    rag = RagEngine(pdf_path=tmp_path, build_index=True)

    st.toast("RAG Engine initialized!")
    print("RagEngine created and index built.")

    # Input for user's question
    user_question = st.text_input("Your Question", placeholder="Ask something about the PDF...")

    if user_question:
        with st.spinner("Thinking..."):
            answer, evidences = rag.query(user_question, return_evidence=True)
            st.markdown("### Answer")
            st.write(answer)

            if evidences:
                st.markdown("### Evidence")
                for ev in evidences:
                    st.write(f"Page: {ev['page']} — Score: {ev['score']:.4f}")
                    st.write(ev['text'])
                    st.markdown("---")
