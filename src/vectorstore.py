"""
Builds / loads the Qdrant vector store (real vector DB, not an in-memory index).

Why Qdrant over FAISS: FAISS is a library, not a database -- it has no
persistence server, no concurrent-write story, and no cloud-hosted option
you can point a deployed app at. Qdrant runs as its own service (via Docker
locally, or Qdrant Cloud in production), which is what "production shape"
actually means: the vector store outlives the process and can be shared by
multiple app instances.
"""
from typing import List

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from .config import RAGConfig


def get_embeddings(cfg: RAGConfig) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=cfg.embedding_model)


def _get_client(cfg: RAGConfig) -> QdrantClient:
    return QdrantClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key or None,
    )


def build_vectorstore(chunks: List[Document], cfg: RAGConfig) -> QdrantVectorStore:
    embeddings = get_embeddings(cfg)
    client = _get_client(cfg)

    # (Re)create the collection so re-running ingest doesn't duplicate points.
    if client.collection_exists(cfg.qdrant_collection):
        client.delete_collection(cfg.qdrant_collection)
    client.create_collection(
        collection_name=cfg.qdrant_collection,
        vectors_config=VectorParams(size=cfg.embedding_dim, distance=Distance.COSINE),
    )

    store = QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key or None,
        collection_name=cfg.qdrant_collection,
    )
    print(f"[vectorstore] indexed {len(chunks)} chunks into Qdrant "
          f"collection '{cfg.qdrant_collection}' at {cfg.qdrant_url}")
    return store


def load_vectorstore(cfg: RAGConfig) -> QdrantVectorStore:
    embeddings = get_embeddings(cfg)
    client = _get_client(cfg)
    if not client.collection_exists(cfg.qdrant_collection):
        raise FileNotFoundError(
            f"Qdrant collection '{cfg.qdrant_collection}' not found at {cfg.qdrant_url}. "
            "Run `python -m src.ingest` first (and make sure Qdrant is running: "
            "`docker compose up -d qdrant`)."
        )
    return QdrantVectorStore(
        client=client,
        collection_name=cfg.qdrant_collection,
        embedding=embeddings,
    )
