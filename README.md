# RAG Research Assistant

A document Q&A system built with **LangChain + Gemini + Qdrant**, with
**hybrid retrieval (BM25 + dense embeddings, fused via Reciprocal Rank
Fusion)**, multi-turn conversation memory, a **FastAPI** backend, a
**Streamlit** frontend, **Docker**-based local deployment, and a
**RAGAS**-based evaluation harness measuring faithfulness, relevancy, and
retrieval precision/recall separately.

## Why this isn't a tutorial clone

- **Evaluation, not vibes.** `evaluation/eval_ragas.py` runs RAGAS's
  faithfulness / answer_relevancy / context_precision / context_recall
  metrics, so retrieval quality and generation quality are measured
  separately instead of eyeballing a demo. See [Evaluation](#evaluation).
- **Hybrid retrieval is the headline technical choice.** Dense embeddings
  alone miss exact keyword/ID matches; BM25 alone misses paraphrases and
  synonyms. This system runs both and fuses the rankings with RRF instead
  of picking one. See [Why hybrid search](#why-hybrid-search-bm25--dense-embeddings).
- **Production shape.** A real vector database (Qdrant, not an in-memory
  index), a FastAPI backend decoupled from the UI, structured
  latency/request logging, and a deployable architecture (Streamlit Cloud
  + Render + Qdrant Cloud).

## Architecture

```
 documents (.txt/.md/.pdf)
        │
        ▼
  chunking (recursive / semantic / markdown)
        │
        ├──────────────────────┐
        ▼                      ▼
  Gemini embeddings       BM25 tokenization
        │                      │
        ▼                      ▼
  Qdrant (dense index)    local BM25 index (pickled)
        │                      │
        └──────── query ───────┘
                    │
                    ▼
         Reciprocal Rank Fusion
                    │
                    ▼
       top-k fused, relevant chunks
                    │
                    ▼
    history-aware query rewriting (multi-turn memory)
                    │
                    ▼
        grounded-answer prompt → Gemini
                    │
                    ▼
           FastAPI /chat endpoint
                    │
                    ▼
         Streamlit UI (separate process)
```

```
┌─────────────────┐      HTTP       ┌──────────────────┐      ┌─────────────┐
│  Streamlit UI    │ ───────────────▶│  FastAPI backend  │─────▶│   Qdrant    │
│ (Streamlit Cloud)│◀─────────────── │    (Render)       │      │   Cloud     │
└─────────────────┘                 └──────────────────┘      └─────────────┘
```

## Why hybrid search (BM25 + dense embeddings)

Dense embeddings capture *meaning* — "car" and "automobile" land near each
other in vector space — but they can miss exact matches on rare tokens:
error codes, product SKUs, acronyms, proper nouns the embedding model
under-weighted. BM25 is the reverse: pure term-frequency ranking, which
nails exact keyword overlap but has zero notion of synonymy.

Instead of a weighted blend of BM25 and cosine scores (which live on
incompatible numeric scales and require fragile calibration), this system
fuses on **rank position** using Reciprocal Rank Fusion:

```
RRF_score(doc) = sum over each retriever r of  1 / (k + rank_r(doc))
```

A document that both retrievers rank highly wins; a document that's
mediocre-but-present in both often outranks a document that's #1 in only
one retriever — which is the behavior you want from hybrid search: broad
consensus beats a single retriever's blind spot. `k=60` is the standard
RRF smoothing constant from the original paper (Cormack et al., 2009).

**Trade-off to be upfront about:** hybrid search costs two retrieval calls
instead of one, and BM25 needs its own index maintained alongside the
vector DB. For a small, fairly homogeneous corpus, dense-only retrieval
might perform nearly as well. Hybrid earns its keep on corpora with a mix
of narrative text and exact identifiers (support tickets, technical docs
with error codes, legal text with case numbers) — worth stating plainly if
asked in an interview why you didn't just add reranking instead.

## Setup

```bash
git clone <your-repo-url>
cd research_assistant
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
cp .env.example .env             # add your GOOGLE_API_KEY
```

## Run locally with Docker (recommended)

```bash
docker compose up -d          # starts Qdrant + the FastAPI API
docker compose exec api python -m src.ingest   # build the hybrid index
```

Then run the Streamlit UI (outside Docker, so `streamlit run` hot-reloads):

```bash
streamlit run app.py
```

## Run locally without Docker

```bash
# 1. Start Qdrant (needs Docker for this one piece even without full compose)
docker run -p 6333:6333 qdrant/qdrant

# 2. Build the hybrid index (dense -> Qdrant, sparse -> local BM25 pickle)
python -m src.ingest

# 3a. Terminal chat (talks to the pipeline directly, no API needed)
python cli.py

# 3b. OR run the API + Streamlit UI as two processes
uvicorn api.main:app --reload
streamlit run app.py   # in a second terminal
```

## Deployment

| Component | Where | Notes |
|---|---|---|
| Vector DB | **Qdrant Cloud** (free tier) | Create a cluster, put its URL + API key in `.env` / your Render env vars |
| API | **Render** (free web service) | Point it at this repo, build command `pip install -r requirements.txt`, start command `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| UI | **Streamlit Community Cloud** | Deploy `app.py`; set `API_BASE_URL` in the app's Secrets to your Render URL |

Streamlit Cloud can only run the Streamlit process itself — it can't host
a separate FastAPI service or a database container, which is why the API
and vector DB live elsewhere and the UI just calls them over HTTP.

## Evaluation

Two harnesses, for two different levels of rigor:

**`evaluation/eval_qa.py`** — lightweight custom LLM-as-judge, scores
answer relevance (1–5) and groundedness (yes/no hallucination check).
Good for a quick sanity check after a config change.

**`evaluation/eval_ragas.py`** — [RAGAS](https://github.com/explodinggradients/ragas),
the standard RAG evaluation library, scoring four metrics *separately*:

| Metric | What it isolates |
|---|---|
| `faithfulness` | Hallucination — is the answer grounded in retrieved context |
| `answer_relevancy` | Does the answer address the actual question |
| `context_precision` | Of retrieved chunks, how many were relevant (retrieval, not generation) |
| `context_recall` | Of the relevant info that exists, how much did retrieval surface |

Separating precision/recall from faithfulness/relevancy is what lets you
diagnose *why* an answer is bad — retrieval missed the right chunk, or the
LLM ignored good context it was given. That distinction is the whole point
of evaluating a RAG system properly instead of eyeballing outputs.

```bash
python -m evaluation.eval_ragas
```

Results are written to `evaluation/results/ragas_last_run.csv` and
`ragas_summary.json`. Rerun after any retrieval/chunking change and update
the table below with real numbers before citing them anywhere — including
a resume.

| Config | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|:---:|:---:|:---:|:---:|
| dense-only baseline | fill in after running | | | |
| hybrid (BM25 + dense, RRF) | fill in after running | | | |

## Observability

The FastAPI backend logs structured JSON lines to stdout for every
request: request ID, path, status code, and latency; `/chat` calls
additionally log question/answer length, chunks retrieved, and end-to-end
latency. In a real deployment this ships to CloudWatch / Datadog / Grafana
Loki unchanged — JSON-lines-to-stdout is the standard log-shipping
contract, not a toy format.

## Project structure

```
research_assistant/
├── app.py                     # Streamlit UI (HTTP client, deployable standalone)
├── cli.py                     # Terminal chat (direct import, for local dev)
├── Dockerfile                 # FastAPI backend image
├── docker-compose.yml         # API + Qdrant, one command
├── api/
│   ├── main.py                  # FastAPI app: /chat, /reset, /health + logging
│   └── schemas.py               # Pydantic request/response models
├── src/
│   ├── config.py                # All tunable parameters (retrieval, models, DB)
│   ├── document_loader.py       # .txt/.md/.pdf loading
│   ├── chunking.py               # Recursive / semantic / markdown splitters
│   ├── vectorstore.py            # Qdrant build/load (dense index)
│   ├── bm25_index.py             # BM25 build/load (sparse index)
│   ├── hybrid_retriever.py       # RRF fusion of dense + sparse results
│   ├── ingest.py                  # Ingestion entrypoint (builds both indexes)
│   └── rag_chain.py              # History-aware retrieval + memory + generation
├── evaluation/
│   ├── test_questions.json       # Eval question set
│   ├── eval_qa.py                 # Lightweight custom LLM-as-judge
│   └── eval_ragas.py              # RAGAS: faithfulness/relevancy/precision/recall
└── data/sample_docs/              # Example knowledge base
```

## Talking points for an interview

- **Why RRF over a weighted score blend:** BM25 and cosine similarity
  scores aren't on comparable scales; fusing on rank avoids needing to
  calibrate a blend weight per corpus.
- **Why Qdrant over FAISS:** FAISS is a library with no server, no
  persistence across process restarts by default, and no cloud-hosted
  option — Qdrant is an actual database you can deploy independently of
  the app.
- **Why separate the API from the UI:** lets each scale/deploy
  independently, and is the only way to put the UI on Streamlit Cloud
  while keeping a real backend.
- **What I'd add next:** cross-encoder reranking on top of the RRF-fused
  candidates (cheap accuracy win before the LLM ever sees the context),
  and query classification to skip hybrid search entirely for queries that
  are clearly keyword lookups vs. semantic questions.
