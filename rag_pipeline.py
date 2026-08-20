"""
rag_pipeline.py – Full RAG pipeline for MediSum.

Pipeline:
  1. Extract text from uploaded PDF files (pdfplumber)
  2. Chunk text with RecursiveCharacterTextSplitter
  3. Embed chunks with FastEmbed BAAI/bge-small-en-v1.5 (ONNX-based, no torch/torchvision needed)
  4. Store in ChromaDB in-memory vectorstore
  5. Retrieve relevant chunks via similarity search
  6. Generate structured medical summary via local Ollama LLM or Groq API (configurable)

LLM Provider:
  Set LLM_PROVIDER=ollama  in .env to use a local Ollama model (default).
  Set LLM_PROVIDER=groq    in .env to use the Groq API (requires GROQ_API_KEY).
"""

import os
import io
import logging

import pdfplumber
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K   = 6   # chunks retrieved per query

# FastEmbed ONNX embedding model – no torch/torchvision needed
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"

# ── LLM Provider settings (read from .env) ─────────────────────────────────────
# Supported: "ollama" (local, default) | "groq" (cloud API)
LLM_PROVIDER     = os.getenv("LLM_PROVIDER", "ollama").lower()

# Ollama settings
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Groq settings (only needed if LLM_PROVIDER=groq)
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama3-8b-8192")


def get_llm():
    """
    Return the configured LLM instance based on LLM_PROVIDER env var.

    - "ollama"  → ChatOllama (local, no API key needed)
    - "groq"    → ChatGroq   (cloud, requires GROQ_API_KEY)
    """
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        # Try Streamlit secrets first, then .env
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except (FileNotFoundError, KeyError):
            api_key = os.getenv("GROQ_API_KEY", "")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file or Streamlit secrets.\n"
                "Get a free key at https://console.groq.com"
            )
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=api_key,
            temperature=0.2,
            max_tokens=2048,
        )

    else:  # default: ollama
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
            num_predict=2048,   # max tokens to generate
        )



# ── Structured Medical Prompt Template (ChatPromptTemplate format) ─────────────

SYSTEM_PROMPT = """You are MediSum, an expert AI medical report analyst. \
Your task is to produce a comprehensive, structured summary from the provided medical document content.

Based on the context below, generate a detailed structured report with ALL of the following sections.
For any section where information is not available in the documents, write "Not specified in the provided documents."

Context from medical reports:
{context}

Generate a complete medical summary using EXACTLY this structure:

**Patient Details:**
- Patient Name: [extract or "Not specified"]
- Patient ID: [extract or "Not specified"]
- Age/Sex: [extract or "Not specified"]
- Registration Date: [extract or "Not specified"]
- Test Report Type: [extract the report types]

**Medical History:**
- Summary of relevant past illnesses: [summarize]
- Chronic conditions: [list any]
- Significant medical events (surgeries, hospitalizations): [mention if any]

**Symptoms and Diagnosis:**
- Primary symptoms: [summarize]
- Onset and progression: [describe]
- Diagnosis made: [state the diagnosis]

**Treatment and Recommendations:**
- Medications administered: [list]
- Therapies or procedures: [summarize]
- Follow-up care recommended: [mention]
- Lifestyle adjustments advised: [include if any]

**Lab Reports:**
- Key findings from blood tests / scans / urinalysis: [summarize]
- Significant results or abnormalities: [highlight]
- Reference ranges and deviations: [note any]

**Current Status:**
- Current condition: [describe]
- Progress or ongoing issues: [mention]
- Prognosis (if mentioned): [state]

Keep the summary structured, professional, and within 1500 words. \
Use plain medical language that is clear to both healthcare professionals and patients."""

# ── Cached Resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI embedding model…")
def get_embeddings() -> FastEmbedEmbeddings:
    """
    Load FastEmbed ONNX embeddings once and cache.
    Uses ONNX runtime – no PyTorch or torchvision required.
    Model is downloaded once (~130 MB) and cached locally.
    """
    return FastEmbedEmbeddings(
        model_name=EMBED_MODEL,
        # Cache models in a local directory to avoid re-downloading
    )



def _get_groq_api_key() -> str:
    """Get Groq API key from Streamlit secrets or environment."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, KeyError):
        key = os.getenv("GROQ_API_KEY", "")
        return key


# ── Step 1: PDF Text Extraction ───────────────────────────────────────────────

def extract_text_from_pdfs(uploaded_files: list) -> list[Document]:
    """
    Extract text from a list of Streamlit UploadedFile objects using pdfplumber.

    Returns a list of LangChain Document objects (one per page).
    """
    documents: list[Document] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        filename   = uploaded_file.name

        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        doc = Document(
                            page_content=text.strip(),
                            metadata={
                                "source":   filename,
                                "page":     page_num,
                                "filename": filename,
                            },
                        )
                        documents.append(doc)
        except Exception as exc:
            logger.warning("Could not extract text from %s: %s", filename, exc)

        # Reset pointer so the file can be re-read if needed
        uploaded_file.seek(0)

    logger.info("Extracted %d pages from %d files", len(documents), len(uploaded_files))
    return documents


# ── Step 2: Text Chunking ──────────────────────────────────────────────────────

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller overlapping chunks for better retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Created %d chunks from %d documents", len(chunks), len(documents))
    return chunks


# ── Step 3 & 4: Embedding + ChromaDB ─────────────────────────────────────────

def build_vectorstore(chunks: list[Document]) -> Chroma:
    """
    Embed chunks and build an in-memory ChromaDB vectorstore.
    Returns the vectorstore object.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    return vectorstore


# ── Step 5: Retriever ─────────────────────────────────────────────────────────

def get_retriever(vectorstore: Chroma, k: int = RETRIEVAL_K):
    """Return a similarity-search retriever from the vectorstore."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ── Step 6: LLM Summary Generation ───────────────────────────────────────────

def generate_summary(retriever, patient_info: dict) -> str:
    """
    Use the configured LLM (Ollama or Groq) via LangChain LCEL to generate a
    structured medical summary.

    Args:
        retriever:    ChromaDB retriever loaded with the patient's report chunks.
        patient_info: dict with patient context (name, age, etc.)

    Returns:
        str: Structured medical summary text.
    """
    try:
        llm = get_llm()
    except ValueError as e:
        return f"⚠️ **LLM configuration error:**\n\n{e}"

    # Build prompt using ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # Helper to format retrieved docs into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain: retrieve → format → prompt → LLM → parse
    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Build the query
    patient_name = patient_info.get("name", "the patient")
    query = (
        f"Generate a complete structured medical summary for patient {patient_name}. "
        f"Include all sections: Patient Details, Medical History, Symptoms and Diagnosis, "
        f"Treatment and Recommendations, Lab Reports, and Current Status."
    )

    try:
        answer = chain.invoke(query)
        if not answer or not answer.strip():
            return "⚠️ The LLM returned an empty response. Please try again."
        return answer.strip()
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        provider = LLM_PROVIDER.upper()
        hint = (
            f"Make sure Ollama is running (`ollama serve`) and the model "
            f"'{OLLAMA_MODEL}' is pulled (`ollama pull {OLLAMA_MODEL}`)."
            if LLM_PROVIDER == "ollama"
            else "Check your GROQ_API_KEY in the .env file."
        )
        return f"⚠️ **{provider} summary generation failed.**\n\n{hint}\n\nError: {exc}"




# ── Main Entry Point ──────────────────────────────────────────────────────────

def process_reports(
    uploaded_files: list,
    patient_info: dict,
    progress_callback=None,
) -> str:
    """
    Full RAG pipeline: PDF → chunks → embeddings → ChromaDB → Groq → summary.

    Args:
        uploaded_files:    List of Streamlit UploadedFile objects.
        patient_info:      Patient metadata dict.
        progress_callback: Optional callable(step: int, total: int, msg: str)

    Returns:
        str: Structured AI-generated summary.
    """
    steps = 4

    def _progress(step: int, msg: str):
        if progress_callback:
            progress_callback(step, steps, msg)

    _progress(1, "📄 Extracting text from PDFs…")
    documents = extract_text_from_pdfs(uploaded_files)

    if not documents:
        return (
            "⚠️ **No readable text found in the uploaded PDFs.**\n\n"
            "Please ensure your PDFs contain selectable text (not scanned images without OCR)."
        )

    _progress(2, "✂️ Chunking and embedding documents…")
    chunks     = chunk_documents(documents)
    vectorstore = build_vectorstore(chunks)

    _progress(3, "🔍 Retrieving relevant information…")
    retriever  = get_retriever(vectorstore)

    _progress(4, "🤖 Generating AI summary with Groq…")
    summary    = generate_summary(retriever, patient_info)

    return summary
