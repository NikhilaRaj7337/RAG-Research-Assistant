"""
Ingest documents into a FAISS vector index.

Supports three chunking strategies (--strategy):
  - recursive : RecursiveCharacterTextSplitter (default)
  - token     : token-count based splitting via tiktoken
  - semantic  : greedy sentence grouping by embedding-similarity drift

Embeddings are cached on disk keyed by a content hash, so re-running ingest on
unchanged text does not re-call the embeddings API (this is the "embedding
optimization" referenced in the project description).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path("data/embedding_cache.jsonl")


def load_documents(docs_dir: str):
    """Load .txt/.md files from docs_dir into LangChain Document objects."""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.[tm][xd]*",  # .txt / .md
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        show_progress=True,
    )
    docs = loader.load()
    if not docs:
        raise ValueError(
            f"No .txt/.md documents found under {docs_dir}. "
            "Point --docs at a folder containing your source documents."
        )
    return docs


def split_recursive(docs, chunk_size=800, chunk_overlap=120):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def split_token(docs, chunk_size=300, chunk_overlap=40):
    from langchain.text_splitter import TokenTextSplitter

    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def split_semantic(docs, similarity_threshold=0.72, max_sentences_per_chunk=12):
    """
    Greedy semantic chunking: walk sentence by sentence, start a new chunk when
    the embedding similarity between the running chunk centroid and the next
    sentence drops below `similarity_threshold`.

    This is intentionally simple (no external semantic-chunking library) so the
    project has no hidden dependency beyond what's in requirements.txt.
    """
    import re

    import numpy as np
    from langchain.schema import Document

    embeddings = get_embedder()
    out = []
    for doc in docs:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc.page_content) if s.strip()]
        if not sentences:
            continue
        vectors = np.array(embed_texts_cached(embeddings, sentences))
        chunks, current, current_vecs = [], [sentences[0]], [vectors[0]]
        for sent, vec in zip(sentences[1:], vectors[1:]):
            centroid = np.mean(current_vecs, axis=0)
            sim = float(
                np.dot(centroid, vec) / (np.linalg.norm(centroid) * np.linalg.norm(vec) + 1e-8)
            )
            if sim < similarity_threshold or len(current) >= max_sentences_per_chunk:
                chunks.append(" ".join(current))
                current, current_vecs = [sent], [vec]
            else:
                current.append(sent)
                current_vecs.append(vec)
        if current:
            chunks.append(" ".join(current))

        for i, c in enumerate(chunks):
            out.append(Document(page_content=c, metadata={**doc.metadata, "chunk": i}))
    return out


def get_embedder():
    from providers import get_embedder as _get_embedder

    return _get_embedder()


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    cache = {}
    with open(CACHE_PATH) as f:
        for line in f:
            row = json.loads(line)
            cache[row["hash"]] = row["embedding"]
    return cache


def _append_cache(rows):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def embed_texts_cached(embedder, texts, batch_size=64):
    """Embed texts with a persistent on-disk cache keyed by content hash."""
    cache = _load_cache()
    hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]

    to_embed_idx = [i for i, h in enumerate(hashes) if h not in cache]
    for start in range(0, len(to_embed_idx), batch_size):
        batch_idx = to_embed_idx[start : start + batch_size]
        batch_texts = [texts[i] for i in batch_idx]
        if not batch_texts:
            continue
        vectors = embedder.embed_documents(batch_texts)
        new_rows = []
        for i, vec in zip(batch_idx, vectors):
            cache[hashes[i]] = vec
            new_rows.append({"hash": hashes[i], "embedding": vec})
        _append_cache(new_rows)

    return [cache[h] for h in hashes]


def build_index(chunks, index_dir: str):
    from langchain_community.vectorstores import FAISS

    embedder = get_embedder()
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # Use the cache so re-ingesting unchanged content is free.
    vectors = embed_texts_cached(embedder, texts)
    text_embedding_pairs = list(zip(texts, vectors))

    store = FAISS.from_embeddings(text_embedding_pairs, embedder, metadatas=metadatas)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    store.save_local(index_dir)
    print(f"Saved FAISS index with {len(chunks)} chunks to {index_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True, help="Directory of source .txt/.md documents")
    parser.add_argument("--index_dir", required=True, help="Output directory for the FAISS index")
    parser.add_argument(
        "--strategy", choices=["recursive", "token", "semantic"], default="recursive"
    )
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--chunk_overlap", type=int, default=120)
    args = parser.parse_args()

    docs = load_documents(args.docs)
    print(f"Loaded {len(docs)} documents")

    if args.strategy == "recursive":
        chunks = split_recursive(docs, args.chunk_size, args.chunk_overlap)
    elif args.strategy == "token":
        chunks = split_token(docs, args.chunk_size, args.chunk_overlap)
    else:
        chunks = split_semantic(docs)

    print(f"Split into {len(chunks)} chunks using '{args.strategy}' strategy")
    build_index(chunks, args.index_dir)


if __name__ == "__main__":
    main()
