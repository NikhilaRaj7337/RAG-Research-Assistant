"""
Builds the conversational RAG chain: FAISS retriever + chat memory + a
grounded-answer prompt designed to reduce hallucination by forcing the model
to (a) only use retrieved context and (b) explicitly say when it doesn't know.
"""
import os

from dotenv import load_dotenv

load_dotenv()

GROUNDED_SYSTEM_PROMPT = """You are a research assistant answering questions about a \
specific document collection. Follow these rules strictly:

1. Answer ONLY using the information in the provided context below.
2. If the context does not contain enough information to answer, say \
"I don't have enough information in the provided documents to answer that" — \
do not guess or use outside knowledge.
3. When you make a factual claim, it must be traceable to a specific passage in \
the context. Do not combine facts from different sources into a claim that \
isn't directly supported.
4. Be concise and cite which source/section you drew from when possible.

Context:
{context}
"""


def load_index(index_dir: str):
    from langchain_community.vectorstores import FAISS
    from providers import get_embedder

    embedder = get_embedder()
    return FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)


def build_chain(index_dir: str, k: int = 4, temperature: float = 0.0):
    """
    Returns a callable chain object with a `.invoke({"question": ..., "chat_history": [...]})`
    interface and built-in multi-turn memory (the caller doesn't need to manage
    chat_history manually — see RagSession below for the stateful wrapper).
    """
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    from providers import get_chat_model

    store = load_index(index_dir)
    retriever = store.as_retriever(search_kwargs={"k": k})

    llm = get_chat_model(temperature=temperature)

    qa_prompt = PromptTemplate(
        template=GROUNDED_SYSTEM_PROMPT + "\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"],
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )
    return chain


class RagSession:
    """Thin stateful wrapper so app.py / eval scripts can just call .ask(question)."""

    def __init__(self, index_dir: str, k: int = 4, temperature: float = 0.0):
        self.chain = build_chain(index_dir, k=k, temperature=temperature)

    def ask(self, question: str):
        result = self.chain.invoke({"question": question})
        return {
            "answer": result["answer"],
            "source_documents": result.get("source_documents", []),
        }

    def reset(self):
        self.chain.memory.clear()
