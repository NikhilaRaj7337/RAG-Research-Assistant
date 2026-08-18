"""
Multi-turn, memory-aware RAG chain.

Architecture:
  user question + chat history
        -> history-aware retriever (rewrites follow-up Qs into standalone queries)
        -> FAISS retriever (MMR search over chunk embeddings)
        -> context-stuffed prompt, instructed to refuse when context is insufficient
        -> Gemini answer
        -> answer + source chunks returned together (for citation / relevance checks)
"""
from typing import Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from .config import RAGConfig
from .vectorstore import load_vectorstore
from .bm25_index import load_bm25_index
from .hybrid_retriever import HybridRetriever

# Anti-hallucination system prompt: explicitly instructs the model to say
# "I don't know" when the retrieved context doesn't cover the question,
# rather than filling gaps from parametric knowledge.
QA_SYSTEM_PROMPT = (
    "You are a research assistant answering questions using ONLY the "
    "provided context. Follow these rules strictly:\n"
    "1. Base every claim in your answer on the context below.\n"
    "2. If the context does not contain enough information to answer, say "
    "\"I don't have enough information in the provided documents to answer that.\"\n"
    "3. Do not use outside knowledge, even if you are confident about it.\n"
    "4. When you use a fact from a specific source, mention the source file name.\n"
    "5. Be concise and directly answer the question first, then add supporting detail.\n\n"
    "Context:\n{context}"
)

CONDENSE_QUESTION_PROMPT = (
    "Given the chat history and a follow-up question, rewrite the follow-up "
    "into a standalone question that contains all necessary context. "
    "Do NOT answer the question, only rewrite it."
)


class ResearchAssistant:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.vectorstore = load_vectorstore(cfg)
        bm25, bm25_documents = load_bm25_index(cfg)
        self.retriever = HybridRetriever(
            vectorstore=self.vectorstore, bm25=bm25, bm25_documents=bm25_documents, cfg=cfg
        )
        self.llm = ChatGoogleGenerativeAI(model=cfg.chat_model, temperature=cfg.temperature)
        self._session_histories: Dict[str, InMemoryChatMessageHistory] = {}
        self._chain = self._build_chain()

    def _get_retriever(self):
        return self.retriever

    def _build_chain(self):
        retriever = self._get_retriever()

        condense_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONDENSE_QUESTION_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, condense_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        combine_docs_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

        def get_history(session_id: str) -> InMemoryChatMessageHistory:
            if session_id not in self._session_histories:
                self._session_histories[session_id] = InMemoryChatMessageHistory()
            return self._session_histories[session_id]

        return RunnableWithMessageHistory(
            rag_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def ask(self, question: str, session_id: str = "default") -> Dict:
        import time

        t0 = time.perf_counter()
        result = self._chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        total_ms = (time.perf_counter() - t0) * 1000

        sources: List[str] = sorted(
            {d.metadata.get("source", "unknown") for d in result.get("context", [])}
        )

        return {
            "answer": result["answer"],
            "sources": sources,
            "num_chunks_retrieved": len(result.get("context", [])),
            "total_latency_ms": round(total_ms, 1),
        }

    def reset(self, session_id: str = "default"):
        self._session_histories.pop(session_id, None)
