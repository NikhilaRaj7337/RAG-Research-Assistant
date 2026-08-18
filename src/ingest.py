"""
One-off ingestion pipeline: load docs -> chunk -> embed -> save FAISS index.

Usage:
    python -m src.ingest
"""
from .config import CONFIG
from .document_loader import load_documents
from .chunking import chunk_documents
from .vectorstore import build_vectorstore
from .bm25_index import build_bm25_index


def main():
    docs = load_documents(CONFIG.docs_dir)
    chunks = chunk_documents(docs, CONFIG)
    build_vectorstore(chunks, CONFIG)   # dense index -> Qdrant
    build_bm25_index(chunks, CONFIG)    # sparse index -> local pickle
    print("[ingest] done. Hybrid index (dense + sparse) is ready.")


if __name__ == "__main__":
    main()
