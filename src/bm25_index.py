"""
BM25 sparse (keyword) index -- the other half of hybrid search.

Dense embeddings are great at "meaning" similarity but miss exact-term
matches (acronyms, IDs, rare proper nouns, error codes). BM25 is the
opposite: it's a term-frequency ranking function that excels at exact
keyword overlap but has no notion of synonymy. Combining both and fusing
the rankings covers each other's blind spots -- this is the standard
"hybrid search" pattern used in production RAG systems.
"""
import pickle
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .config import RAGConfig


def _tokenize(text: str) -> List[str]:
    # Simple, fast, language-agnostic-enough tokenizer. Swap for a proper
    # tokenizer (e.g. spaCy) if you need stemming/stopword handling.
    return text.lower().split()


def build_bm25_index(chunks: List[Document], cfg: RAGConfig) -> None:
    corpus_tokens = [_tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    out_path = Path(cfg.bm25_index_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": chunks}, f)

    print(f"[bm25_index] built BM25 index over {len(chunks)} chunks -> {out_path}")


def load_bm25_index(cfg: RAGConfig) -> Tuple[BM25Okapi, List[Document]]:
    path = Path(cfg.bm25_index_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No BM25 index at {path}. Run `python -m src.ingest` first."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["documents"]


def bm25_search(bm25: BM25Okapi, documents: List[Document], query: str, k: int) -> List[Document]:
    scores = bm25.get_scores(_tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [documents[i] for i in ranked_idx]
