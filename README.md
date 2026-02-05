# 📘 NCERT Class 8 Science AI Tutor (RAG Project)

This project implements a **Retrieval-Augmented Generation (RAG) based AI Tutor** that answers questions strictly from the NCERT Class 8 Science textbook.

The system uses semantic search and a local language model to generate grounded, syllabus-based answers while preventing hallucinations.

It also provides **source citations** for transparency and includes an interactive **Streamlit chat interface**.

---

# 🚀 Project Overview

Generative AI models often produce answers from general internet knowledge, which can lead to incorrect or out-of-syllabus responses in educational settings.

This project solves that problem by restricting the AI tutor to **NCERT Class 8 Science textbook content only**.

The system retrieves relevant textbook sections first and then generates answers using only that retrieved context.

This ensures:
- Accurate answers  
- Syllabus alignment  
- Reduced hallucination  
- Transparent citations  

---

# 🧠 How the System Works (RAG Pipeline)

The project follows a Retrieval-Augmented Generation workflow:

1. NCERT textbook PDFs are cleaned and converted to text  
2. Text is split into smaller chunks  
3. Each chunk is converted into embeddings (vectors)  
4. Embeddings stored in FAISS vector database  
5. User question converted into embedding  
6. FAISS retrieves most relevant textbook chunks  
7. Local language model generates answer from retrieved text  
8. System displays answer with source citations  

This approach ensures answers remain grounded in textbook knowledge.

---

# 🏗 Tech Stack

**Programming Language**
- Python

**AI / ML**
- Sentence Transformers (all-MiniLM-L6-v2)
- HuggingFace Transformers (FLAN-T5)

**Vector Database**
- FAISS (Facebook AI Similarity Search)

**Frontend**
- Streamlit

**Data Processing**
- Pandas
- NumPy

**Evaluation**
- BLEU score
- ROUGE-L score

---

# 💬 Features

- Curriculum-based AI tutor  
- Retrieval-Augmented Generation pipeline  
- Semantic search using embeddings  
- FAISS vector database  
- Local LLM (no API cost)  
- Out-of-syllabus query handling  
- Source citations for transparency  
- BLEU & ROUGE-L evaluation  
- Streamlit chat interface  

---

# 📊 Evaluation

The system was evaluated using 10 representative textbook questions.

Metrics used:
- BLEU Score
- ROUGE-L Score
- Manual review

BLEU scores remain moderate due to paraphrasing, while ROUGE-L shows strong overlap with reference answers.  
The system consistently generates textbook-grounded responses.

---

# 💻 How to Run Locally

## 1. Clone repository

## 2. Install dependencies

## 3. Run Streamlit app

---

# 🧪 Example Questions to Try

- What is photosynthesis?  
- Why are plants called producers?  
- What is friction?  
- Define force  
- What are herbivores?  

Out-of-syllabus queries will return:
> “I’m focused on Class 8 Science; try re-phrasing.”

---

# 📁 Project Structure
ncert-ai-tutor-rag/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│ └── cleaned/
│ └── class8_science.jsonl
│
├── notebooks/
│ └── ai_tutor_class8.ipynb
│
└── evaluation/
├── evaluation.csv
└── interaction_logs.jsonl


---

# 🔮 Future Improvements

- Use larger open-source LLMs for better answer quality  
- Fine-tune model on curated Q&A pairs  
- Improve retrieval ranking  
- Deploy as web application  
- Add multi-subject support  

---

# 👤 Author

This project was built as a hands-on implementation of a real-world Retrieval-Augmented Generation (RAG) system for educational use.

Focus areas:
- AI/LLM systems  
- Semantic search  
- Practical ML deployment  
- End-to-end project building  

---

# ⭐ If you found this useful

Give this repository a star ⭐ to support the project.
