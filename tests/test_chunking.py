"""
Offline tests that don't require an OpenAI API key/network access — only
exercise the pure text-splitting logic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain.schema import Document
from ingest import split_recursive, split_token


def test_recursive_split_respects_chunk_size():
    long_text = "Sentence one. " * 200
    doc = Document(page_content=long_text, metadata={"source": "test.txt"})
    chunks = split_recursive([doc], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 260 for c in chunks)  # some slack for separators


def test_recursive_split_preserves_metadata():
    doc = Document(page_content="A. B. " * 100, metadata={"source": "test.txt"})
    chunks = split_recursive([doc], chunk_size=50, chunk_overlap=5)
    assert all(c.metadata.get("source") == "test.txt" for c in chunks)


def test_token_split_produces_chunks():
    doc = Document(page_content="word " * 500, metadata={"source": "test.txt"})
    chunks = split_token([doc], chunk_size=50, chunk_overlap=5)
    assert len(chunks) > 1


if __name__ == "__main__":
    test_recursive_split_respects_chunk_size()
    test_recursive_split_preserves_metadata()
    test_token_split_produces_chunks()
    print("All offline chunking tests passed.")
