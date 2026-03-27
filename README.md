# 📄 PDF Bot (GenAI)

An AI-powered chatbot that allows users to interact with PDF documents by asking questions in natural language. Built using Python, this project leverages modern GenAI techniques like embeddings, vector search, and LLM-based retrieval to provide accurate, context-aware answers.

---

## 🚀 Overview

PDF Bot enables users to upload PDF documents and extract meaningful insights through conversational queries.

Instead of manually reading long documents, users can simply ask questions, and the system retrieves relevant information from the PDF and generates precise answers using AI.

This project demonstrates real-world implementation of Retrieval-Augmented Generation (RAG) using LLMs and a scalable vector database (Pinecone).

---

## ✨ Features

- 📄 Upload and process a PDF document  
- 🔍 Extract and split text into manageable chunks  
- 🧠 Generate embeddings for semantic understanding  
- ☁️ Store embeddings using Pinecone vector database  
- 🤖 Ask questions in natural language  
- 📊 Context-aware answers using LLM (Google Gemini)  
- 🌐 Interactive UI using Streamlit  
- ⚡ Fast and scalable semantic search  

---

## 🧰 Tech Stack

- **Language:** Python  
- **LLM:** Google Gemini  
- **Frameworks:** LangChain  
- **Vector Database:** Pinecone  
- **Frontend/UI:** Streamlit  
- **Concept:** Retrieval-Augmented Generation (RAG)  

---

## ⚙️ How It Works

1. User uploads a PDF file  
2. Text is extracted and split into chunks  
3. Each chunk is converted into embeddings  
4. Embeddings are stored in Pinecone vector database  
5. User asks a question  
6. System retrieves the most relevant chunks from Pinecone  
7. LLM generates a context-aware answer  
