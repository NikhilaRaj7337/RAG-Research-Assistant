"""
Loads documents from disk. Supports .txt, .md, .pdf out of the box.
"""
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)

SUPPORTED_LOADERS = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pdf": PyPDFLoader,
}


def load_documents(docs_dir: str) -> List[Document]:
    """Walk `docs_dir` and load every supported file into LangChain Documents."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents: List[Document] = []
    for file_path in sorted(docs_path.rglob("*")):
        if not file_path.is_file():
            continue
        loader_cls = SUPPORTED_LOADERS.get(file_path.suffix.lower())
        if loader_cls is None:
            continue
        try:
            loader = loader_cls(str(file_path))
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = file_path.name
            documents.extend(loaded)
        except Exception as e:
            print(f"[document_loader] skipped {file_path.name}: {e}")

    if not documents:
        raise ValueError(f"No supported documents found in {docs_dir}")

    print(f"[document_loader] loaded {len(documents)} document(s) from {docs_dir}")
    return documents
