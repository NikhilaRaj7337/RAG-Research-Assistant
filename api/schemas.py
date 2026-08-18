from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    session_id: str = Field(default="default", description="Conversation session identifier")


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    num_chunks_retrieved: int
    total_latency_ms: float


class ResetRequest(BaseModel):
    session_id: str = Field(default="default")


class HealthResponse(BaseModel):
    status: str
    chat_model: str
    embedding_model: str
    qdrant_collection: str
