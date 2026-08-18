"""
Hybrid retriever: runs BM25 (sparse/keyword) and Qdrant (dense/semantic)
searches in parallel, then fuses the two ranked lists with Reciprocal Rank
Fusion (RRF) instead of picking one or the other.

Why RRF instead of a weighted score blend: BM25 scores and cosine
similarity scores live on completely different numeric scales, so
averaging them directly is meaningless without careful calibration. RRF
sidesteps that by fusing on RANK POSITION, not raw score:

    RRF_score(doc) = sum over each retriever r that returned doc of
                      1 / (k + rank_r(doc))

A doc ranked #1 by both retrievers wins; a doc that's mediocre-but-present
in both often beats a doc that's #1 in only one, which matches what you
want out of hybrid search: consensus wins over a single retriever's blind
spot.
"""
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from pydantic import Field
from rank_bm25 import BM25Okapi

from .bm25_index import bm25_search
from .config import RAGConfig


def _doc_key(doc: Document) -> str:
    # Chunks don't have stable IDs by default, so key on content + source.
    return f"{doc.metadata.get('source', '')}::{doc.page_content[:120]}"


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]], k: int = 60
) -> List[Document]:
    scores = {}
    doc_lookup = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = _doc_key(doc)
            doc_lookup[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    fused_keys = sorted(scores, key=lambda kk: scores[kk], reverse=True)
    return [doc_lookup[kk] for kk in fused_keys]


class HybridRetriever(BaseRetriever):
    """LangChain-compatible retriever: BM25 + Qdrant, fused with RRF."""

    vectorstore: QdrantVectorStore
    bm25: BM25Okapi
    bm25_documents: List[Document]
    cfg: RAGConfig = Field(exclude=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        dense_results = self.vectorstore.similarity_search(query, k=self.cfg.dense_k)
        sparse_results = bm25_search(self.bm25, self.bm25_documents, query, k=self.cfg.sparse_k)
        fused = reciprocal_rank_fusion([dense_results, sparse_results], k=self.cfg.rrf_k)
        return fused[: self.cfg.top_k]

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)
