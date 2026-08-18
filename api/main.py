"""
FastAPI backend for the RAG research assistant.

Run locally:
    uvicorn api.main:app --reload

Run in Docker:
    docker compose up

This is the piece a deployed Streamlit frontend (or any other client)
talks to over HTTP -- separating the API from the UI is what lets you
deploy the UI on Streamlit Cloud while the API + Qdrant live on Render /
Qdrant Cloud.
"""
import logging
import sys
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.config import CONFIG
from src.rag_chain import ResearchAssistant
from .schemas import ChatRequest, ChatResponse, ResetRequest, HealthResponse

# --- Structured logging (JSON lines) -----------------------------------
# In a real deployment this would ship to something like CloudWatch /
# Datadog / Grafana Loki. For a portfolio project, JSON-lines-to-stdout is
# the right level of effort: it's genuinely structured and greppable, and
# any log shipper can pick it up without code changes.
logger = logging.getLogger("research_assistant")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(_handler)


def log_event(event: str, **fields):
    import json
    record = {"event": event, "timestamp": time.time(), **fields}
    logger.info(json.dumps(record))


app = FastAPI(title="RAG Research Assistant API", version="1.0.0")

# Loosened for a portfolio demo; restrict to your Streamlit Cloud origin
# in a real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_assistant: ResearchAssistant | None = None


def get_assistant() -> ResearchAssistant:
    global _assistant
    if _assistant is None:
        _assistant = ResearchAssistant(CONFIG)
    return _assistant


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    log_event(
        "http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=latency_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        chat_model=CONFIG.chat_model,
        embedding_model=CONFIG.embedding_model,
        qdrant_collection=CONFIG.qdrant_collection,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        assistant = get_assistant()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    t0 = time.perf_counter()
    try:
        result = assistant.ask(req.question, session_id=req.session_id)
    except Exception as e:
        log_event("chat_error", session_id=req.session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    log_event(
        "chat_completed",
        session_id=req.session_id,
        question_length=len(req.question),
        answer_length=len(result["answer"]),
        num_chunks_retrieved=result["num_chunks_retrieved"],
        latency_ms=latency_ms,
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        num_chunks_retrieved=result["num_chunks_retrieved"],
        total_latency_ms=result["total_latency_ms"],
    )


@app.post("/reset")
def reset(req: ResetRequest):
    assistant = get_assistant()
    assistant.reset(req.session_id)
    log_event("session_reset", session_id=req.session_id)
    return {"status": "reset", "session_id": req.session_id}
