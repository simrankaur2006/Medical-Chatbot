# 🩺 Medical Chatbot using RAG (Hugging Face + FAISS + LangChain)

This project is a **Medical Question Answering Chatbot** built using **Retrieval-Augmented Generation (RAG)**.  
It allows users to ask medical questions and get answers **strictly based on uploaded medical PDFs**, reducing hallucinations and improving reliability.

The system uses:
- **Hugging Face LLMs** for natural language understanding
- **FAISS** for fast vector similarity search
- **LangChain (1.x)** for orchestration
- **Streamlit** for an interactive chat-based UI

> ⚠️ This chatbot is for **educational purposes only** and is **not a substitute for professional medical advice**.

---

## 🚀 Features

- 📄 Load and processed multiple medical documents
- ✂️ Intelligent text chunking
- 🔍 Semantic search using FAISS
- 🤖 Hugging Face powered LLM responses
- 💬 Chat-style Streamlit interface


---
## 🧠 Architecture Overview
User (Streamlit UI)
↓
LangChain RAG Pipeline
↓
FAISS Vector Store (PDF embeddings)
↓
Hugging Face LLM (Answer Generation)


---

## 📂 Project Structure

Medical-Chatbot/
├── connect_memory_with_llm.py # Core RAG logic (LLM + FAISS)

├── medibot.py # Streamlit UI

├── connect_memory_with_llm.py # PDF ingestion & FAISS creation

├── requirements.txt # Project dependencies

├── .env.example # Environment variable template

├── data/ # Medical PDF files

│ └── *.pdf

├── vectorstore/

│ └── db_faiss/ # FAISS index (generated)

└── README.md


---

## 🛠️ Technologies Used

- **Python 3.9+**
- **LangChain 1.2.x**
- **Hugging Face Inference API**
- **FAISS**
- **Sentence Transformers**
- **Streamlit**

---

## 🔑 Prerequisites

- Python installed
- Git installed
- Hugging Face account & API token

---

## 📥 How to Clone the Project

```bash
git clone https://github.com/<your-username>/Medical-Chatbot.git
cd Medical-Chatbot
pip install -r requirements.txt
python connect_memory_with_llm.py
streamlit run medibot.py



