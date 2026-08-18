"""
Central configuration for the RAG pipeline.
Tweak these values to reproduce / improve the reported metrics.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

@dataclass
class RAGConfig:
    # --- LLM ---
    chat_model: str = "gemini-3.6-flash"   # current GA flash model
    temperature: float = 0.0               # low temp -> fewer hallucinations

    # --- Embeddings ---
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = 3072              # gemini-embedding-001 default output size

    # --- Chunking strategy ---
    chunk_size: int = 800
    chunk_overlap: int = 150
    splitter: str = "recursive"         # "recursive" | "semantic" | "markdown"

    # --- Retrieval (hybrid: BM25 + dense vector, fused with Reciprocal Rank Fusion) ---
    top_k: int = 4                      # final number of chunks passed to the LLM
    dense_k: int = 15                   # candidates pulled from Qdrant before fusion
    sparse_k: int = 15                  # candidates pulled from BM25 before fusion
    rrf_k: int = 60                     # RRF smoothing constant (standard default)

    # --- Memory ---
    memory_window: int = 6              # number of past turns kept verbatim

    # --- Vector DB (Qdrant) ---
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")  # needed for Qdrant Cloud
    qdrant_collection: str = "research_assistant_chunks"

    # --- BM25 sparse index (persisted alongside the vector DB) ---
    bm25_index_path: str = "bm25_index/bm25_index.pkl"

    # --- Paths ---
    docs_dir: str = "data/sample_docs"

    # --- API (used by the Streamlit client to reach the FastAPI backend) ---
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")

CONFIG = RAGConfig()
