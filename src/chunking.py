"""
Chunking strategies. This is the piece that most directly drives
retrieval precision and hallucination rate, so a few strategies are
implemented and swappable via config.splitter.
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from .config import RAGConfig


def _recursive_split(docs: List[Document], cfg: RAGConfig) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def _markdown_split(docs: List[Document], cfg: RAGConfig) -> List[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    chunks: List[Document] = []
    for d in docs:
        for c in header_splitter.split_text(d.page_content):
            c.metadata.update(d.metadata)
            chunks.append(c)
    # further split any oversized sections
    return _recursive_split(chunks, cfg)


def _semantic_split(docs: List[Document], cfg: RAGConfig) -> List[Document]:
    """
    Lightweight 'semantic' chunking: split on paragraph boundaries first,
    then greedily merge paragraphs until chunk_size is reached, so chunks
    stay topically coherent instead of cutting mid-idea.
    """
    chunks: List[Document] = []
    for d in docs:
        paragraphs = [p.strip() for p in d.page_content.split("\n\n") if p.strip()]
        buf, buf_len = [], 0
        for p in paragraphs:
            if buf_len + len(p) > cfg.chunk_size and buf:
                chunks.append(Document(page_content="\n\n".join(buf), metadata=d.metadata))
                buf, buf_len = [], 0
            buf.append(p)
            buf_len += len(p)
        if buf:
            chunks.append(Document(page_content="\n\n".join(buf), metadata=d.metadata))
    return chunks


STRATEGIES = {
    "recursive": _recursive_split,
    "markdown": _markdown_split,
    "semantic": _semantic_split,
}


def chunk_documents(docs: List[Document], cfg: RAGConfig) -> List[Document]:
    strategy = STRATEGIES.get(cfg.splitter, _recursive_split)
    chunks = strategy(docs, cfg)
    print(f"[chunking] strategy={cfg.splitter} produced {len(chunks)} chunks "
          f"(chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap})")
    return chunks
