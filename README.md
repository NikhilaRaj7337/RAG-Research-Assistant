# RAG Research Assistant

A document Q&A agent built with **LangChain + Gemini (swappable to GPT-4) + FAISS**, featuring
multi-turn conversational memory, three configurable chunking strategies, and an evaluation
harness for measuring answer relevance, retrieval precision, and hallucination rate against a
labeled test set.

## Architecture

```
             ┌─────────────┐      ┌─────────────────┐      ┌─────────────┐
  docs/  ──▶ │  ingest.py  │ ──▶  │  FAISS vector    │ ◀──  │  chains.py  │
             │ (chunking + │      │  store (local)   │      │ (retrieval  │
             │ embeddings) │      └─────────────────┘      │  + memory)  │
             └─────────────┘                                └──────┬──────┘
                                                                     │
                                                              ┌──────▼──────┐
                                                              │   app.py    │
                                                              │ (CLI chat)  │
                                                              └─────────────┘
```

- **`src/ingest.py`** — loads `.txt`/`.md` documents, splits them with a configurable chunking
  strategy (`recursive` / `token` / `semantic`), embeds the chunks (with a disk cache so
  re-ingesting unchanged text skips the embeddings API), and persists a FAISS index to disk.
- **`src/chains.py`** — builds a `ConversationalRetrievalChain` combining the FAISS retriever,
  buffered chat memory (multi-turn), and a grounded-answer prompt that instructs the model to
  answer only from retrieved context and say "I don't know" otherwise.
- **`src/providers.py`** — a single provider/factory abstraction deciding Gemini vs. GPT-4 for
  both the chat model and the embedding model, driven by `LLM_PROVIDER` in `.env`.
- **`src/app.py`** — a CLI chat loop against the chain (`exit` to quit, `reset` to clear memory).
- **`eval/evaluate.py`** — runs a labeled QA test set (`eval/test_set.jsonl`) through the chain
  and reports three metrics, averaged across the set:
  - **Answer relevance** — cosine similarity between the generated answer's embedding and a
    reference answer's embedding.
  - **Retrieval precision@k** — whether the labeled gold-source document appears among the
    top-k retrieved chunks.
  - **Hallucination rate** — an LLM grader decomposes the answer into individual factual claims
    and reports the fraction not directly supported by the retrieved context.

This is a deliberately scoped, single-machine project: no hosted vector database, no web
server, no hybrid retrieval, no third-party evaluation library. Everything above actually runs
end-to-end locally with the dependencies pinned in `requirements.txt`.

## Setup

This project defaults to **Google's Gemini API free tier** (no credit card required) so you can
run it end-to-end at zero cost.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env   # Windows: Copy-Item .env.example .env
```

Get a free key at **https://aistudio.google.com/apikey**, then put it in `.env`:
```
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-key-here
GEMINI_CHAT_MODEL=gemini-3.5-flash-lite
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
```

Note: Google rotates/deprecates specific model names periodically. If a model name above 404s,
query your own key's available models and swap in whatever's current:
```bash
curl "https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_KEY" | grep -A2 generateContent
```

Want to use OpenAI/GPT-4 instead (paid)? Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` in
`.env` — no code changes needed, the provider is selected centrally in `src/providers.py`.

## Usage

```bash
# 1. Build the index from documents in data/sample_docs (swap in your own docs)
python src/ingest.py --docs data/sample_docs --index_dir data/faiss_index

# 2. Chat with the assistant
python src/app.py --index_dir data/faiss_index

# 3. Evaluate against a labeled test set
python eval/evaluate.py --index_dir data/faiss_index --test_set eval/test_set.jsonl --k 4
```

## Results

Measured on the 4-question labeled test set in `eval/test_set.jsonl`, against the 2-document
sample corpus in `data/sample_docs/`, using `gemini-3.5-flash-lite`:

| Metric | Score |
|---|---|
| Answer relevance (mean cosine similarity vs. reference) | 0.840 |
| Retrieval precision@4 | 1.000 |
| Hallucination rate (fraction of unsupported claims) | 0.000 |

**On the perfect retrieval precision@4**: the sample corpus has only two source documents, so a
perfect precision@4 here is expected rather than a strong claim — with just two possible source
files, the correct one showing up in the top 4 retrieved chunks isn't a hard bar to clear. This
metric would be far more meaningful measured against a larger document set with more plausible
distractor chunks; treat it as "the pipeline works correctly on a small known set," not as
evidence of retrieval quality at scale.

**On the 0% hallucination rate**: measured on this same 4-question labeled test set — a result
worth citing, but phrase it as "0% hallucination rate on a 4-question labeled test set" rather
than an unqualified "0% hallucination" claim. It's also graded by an LLM decomposing the answer
into claims and checking each against retrieved context, not a human-verified ground truth, so
treat it as a strong signal rather than an absolute guarantee.

Rerun `eval/evaluate.py` after any chunking/retrieval change and update this table with the real
output before citing these numbers anywhere, including a resume.

## Chunking strategies supported (`--strategy`)

| Strategy | Description |
|---|---|
| `recursive` (default) | Recursive character splitting with overlap, respects paragraph/sentence boundaries first |
| `token` | Token-count based splitting (tiktoken), useful for tight context budgets |
| `semantic` | Embedding-similarity based splitting — groups sentences until semantic drift exceeds a threshold |

## Embedding optimization

- Embeddings are cached to disk (`data/embedding_cache.jsonl`) keyed by content hash, so
  re-ingesting unchanged documents doesn't re-call the embeddings API.
- Batch embedding calls to reduce API round trips.

## What this project does not (yet) include

Being upfront about scope rather than overselling it:
- **No hosted vector database.** FAISS is a local, in-process library — adding Qdrant (or
  Pinecone/Weaviate) would mean a real server, metadata filtering, and horizontal scaling, none
  of which this project needs at its current size.
- **No hybrid (BM25 + dense) retrieval.** Dense embedding similarity only. Adding a BM25 index
  alongside FAISS and fusing the two ranked lists via Reciprocal Rank Fusion is a natural next
  step for queries involving exact identifiers (error codes, product IDs) that dense embeddings
  handle poorly.
- **No RAGAS.** `eval/evaluate.py` is a hand-rolled evaluator; swapping in the real `ragas`
  library would add faithfulness/context-precision/context-recall as separate, standardized
  metrics instead of this project's three custom ones.
- **No web server or frontend.** `app.py` is a terminal CLI. A FastAPI backend plus a simple
  frontend would be the natural next step to make this demo-able as a live product rather than
  a script.

## License

MIT
