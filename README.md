# NCERT Class 8 Science – AI Tutor

This project implements a Retrieval-Augmented Generation (RAG) based AI tutor
for the NCERT Class 8 Science syllabus.

## Features
- Semantic search using FAISS
- Local LLM (FLAN-T5) for answer generation
- Textbook-grounded answers
- Source citations for transparency
- BLEU and ROUGE-L evaluation
- Streamlit-based chat interface

## How to Run
1. Create virtual environment
2. Install requirements:
   pip install -r requirements.txt
3. Run the app:
   streamlit run app.py

## Notes
- The system is restricted to NCERT Class 8 Science content.
- Out-of-syllabus queries are handled gracefully.
