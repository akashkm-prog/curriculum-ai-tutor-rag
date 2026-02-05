import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="NCERT Class 8 Science – AI Tutor",
    page_icon="📘",
    layout="centered"
)

st.title("📘 NCERT Class 8 Science – AI Tutor")
st.write("Ask questions strictly from the NCERT Class 8 Science textbook.")

# ---------------------------------
# Load & cache RAG components
# ---------------------------------
@st.cache_resource
def load_rag_components():
    # Load cleaned dataset
    df = pd.read_json("data/cleaned/class8_science.jsonl", lines=True)

    # Chunking function
    def chunk_text(text, chunk_size=300, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunks.append(" ".join(words[i:i + chunk_size]))
        return chunks

    chunks = []
    chunk_meta = []

    for _, row in df.iterrows():
        chapter = row["chapter"]
        text_chunks = chunk_text(row["text"])
        for i, chunk in enumerate(text_chunks):
            chunks.append(chunk)
            chunk_meta.append({
                "chapter": chapter,
                "chunk_id": i
            })

    # Embedding model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(chunks, show_progress_bar=False)

    # FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Local LLM (FREE)
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=256
    )

    return chunks, chunk_meta, embed_model, index, llm


chunks, chunk_meta, embed_model, index, llm = load_rag_components()

# ---------------------------------
# Search function
# ---------------------------------
def search(query, top_k=2):
    q_emb = embed_model.encode([query])
    _, indices = index.search(q_emb, top_k)

    results = []
    sources = []

    for idx in indices[0]:
        results.append(chunks[idx])
        sources.append(chunk_meta[idx])

    return results, sources

# ---------------------------------
# Prompt
# ---------------------------------
def build_prompt(context, question):
    return f"""
You are an AI tutor for NCERT Class 8 Science.
Answer ONLY using the textbook content below.
Use simple, student-friendly language.
If the answer is not present, reply exactly:
"I’m focused on Class 8 Science; try re-phrasing."

Textbook content:
{context}

Question:
{question}

Answer:
"""

# ---------------------------------
# RAG Answer function
# ---------------------------------
def rag_answer(question):
    retrieved_chunks, sources = search(question)
    context = "\n\n".join(retrieved_chunks)
    prompt = build_prompt(context, question)

    answer = llm(prompt)[0]["generated_text"]
    return answer, sources

# ---------------------------------
# Session state
# ---------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------
# Chat input (ALWAYS visible)
# ---------------------------------
user_question = st.chat_input("Ask a Class 8 Science question")

if user_question:
    with st.spinner("Thinking..."):
        answer, sources = rag_answer(user_question)

    st.session_state.chat_history.append({
        "question": user_question,
        "answer": answer,
        "sources": sources
    })

# ---------------------------------
# Display chat history
# ---------------------------------
for chat in reversed(st.session_state.chat_history):
    st.markdown("### 🧑‍🎓 Question")
    st.write(chat["question"])

    st.markdown("### 🤖 Answer")
    st.write(chat["answer"])

    st.markdown("### 📚 Sources")
    for s in chat["sources"]:
        st.write(f"- Chapter: {s['chapter']} | Chunk ID: {s['chunk_id']}")

    st.markdown("---")

