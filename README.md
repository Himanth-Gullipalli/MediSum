🩺 MediSum – AI-Powered Medical Report Summarization System

MediSum is an AI-powered medical report summarization application designed to simplify and organize information from multiple medical documents. The system allows users to upload medical reports such as doctor reports, blood reports, and scan reports, processes their content, and generates a structured and easy-to-understand medical summary.

The project uses a Retrieval-Augmented Generation (RAG) pipeline to extract relevant information from uploaded PDF documents and generate context-aware summaries. Medical documents are converted into vector embeddings and stored in ChromaDB for efficient semantic retrieval. Patient information and generated summaries are managed using MongoDB.

The application provides an interactive Streamlit interface for patient management, report uploads, summary generation, viewing previously generated summaries, and sharing summaries with medical experts through email integration. It also supports automated PDF generation for storing and sharing medical summaries.

Note: MediSum is designed to assist with organizing and summarizing medical information and is not intended to replace professional medical advice or diagnosis.

✨ Key Features
📄 Upload and process multiple medical PDF reports
🔍 Semantic search and context retrieval using a RAG pipeline
🤖 AI-powered generation of structured medical summaries
👤 Patient record management
🧠 Vector-based document retrieval using embeddings
🗄️ Storage of patient information and generated summaries
📑 Automated medical summary PDF generation
📧 Share generated summaries with medical experts
💻 Interactive and user-friendly Streamlit interface
🛠️ Technologies Used

Programming Language

Python

Frontend / Application Framework

Streamlit

AI & RAG Framework

LangChain
Ollama / Local LLM Inference

Vector Database & Embeddings

ChromaDB
Hugging Face Embeddings

Database

MongoDB
PyMongo

Document Processing

PyPDF
LangChain PDF Loaders

PDF Generation

ReportLab

Email Integration

SMTP
Python smtplib

Configuration & Utilities

Python Dotenv
Environment Variables (.env)