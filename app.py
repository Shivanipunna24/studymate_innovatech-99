import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# -------------------------------
# 1. Hugging Face Client (IBM Granite)
# -------------------------------
HF_TOKEN = "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"  # <-- Replace with your Hugging Face token
MODEL_ID = "ibm-granite/granite-3.3-2b-instruct"

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# -------------------------------
# 2. Utility: PDF Text Extraction
# -------------------------------
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# -------------------------------
# 3. Chunking
# -------------------------------
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# -------------------------------
# 4. Embeddings + FAISS Index
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve_similar_chunks(question, chunks, index, embeddings, top_k=3):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]]

# -------------------------------
# 5. Call IBM Granite Model
# -------------------------------
def ask_ibm_granite(question, context):
    prompt = f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = client.text_generation(prompt, max_new_tokens=200)
    return response

# -------------------------------
# 6. Streamlit UI
# -------------------------------
st.set_page_config(page_title="StudyMate - AI PDF Assistant", layout="wide")
st.title("📘 StudyMate: AI-Powered Academic Assistant")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    index, embeddings = build_faiss_index(chunks)
    st.success("PDF processed successfully! You can now ask questions.")

    question = st.text_input("Ask a question about your document:")

    if question:
        relevant_chunks = retrieve_similar_chunks(question, chunks, index, embeddings)
        context = "\n\n".join(relevant_chunks)
        answer = ask_ibm_granite(question, context)
        
        st.subheader("Answer:")
        st.write(answer)

        with st.expander("🔍 Retrieved Context"):
            st.write(context)
