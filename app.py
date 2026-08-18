"""
Streamlit UI -- a thin HTTP client for the FastAPI backend.

This is deliberately NOT importing src/ directly. Streamlit Community Cloud
can only run this one process, so the UI talks to the API over HTTP just
like any other client would. Point API_BASE_URL at:
  - http://localhost:8000            when running the API locally
  - your Render URL                  once deployed

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub, deploy on share.streamlit.io, and set the API_BASE_URL
    secret in the Streamlit Cloud app settings to your Render API URL.
"""
import os
import uuid

import requests
import streamlit as st

API_BASE_URL = st.secrets.get("API_BASE_URL", os.getenv("API_BASE_URL", "http://localhost:8000"))

st.set_page_config(page_title="RAG Research Assistant", page_icon="📚")
st.title("📚 RAG Research Assistant")
st.caption("LangChain + Gemini + Qdrant · hybrid search (BM25 + dense) · multi-turn memory")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []


def call_api(path: str, payload: dict) -> dict:
    resp = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


# Health check so a broken backend fails loudly and clearly, not silently.
try:
    health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
except Exception:
    st.error(f"Can't reach the API at {API_BASE_URL}. Is the backend running?")
    st.info("Locally: `uvicorn api.main:app --reload` in another terminal, or `docker compose up`.")
    st.stop()

for role, content, sources, latency in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            st.caption(f"Sources: {', '.join(sources)}" + (f" · {latency:.0f}ms" if latency else ""))

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.history.append(("user", question, [], None))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                result = call_api(
                    "/chat", {"question": question, "session_id": st.session_state.session_id}
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
                st.stop()
            st.markdown(result["answer"])
            caption = f"Sources: {', '.join(result['sources'])}" if result["sources"] else "No sources retrieved"
            caption += f" · {result['total_latency_ms']:.0f}ms · {result['num_chunks_retrieved']} chunks"
            st.caption(caption)
    st.session_state.history.append(
        ("assistant", result["answer"], result["sources"], result["total_latency_ms"])
    )

with st.sidebar:
    st.subheader("Backend")
    st.write(f"**API:** {API_BASE_URL}")
    st.write(f"**Chat model:** {health['chat_model']}")
    st.write(f"**Embedding model:** {health['embedding_model']}")
    st.write(f"**Qdrant collection:** {health['qdrant_collection']}")
    st.write("**Retrieval:** hybrid (BM25 + dense, RRF-fused)")
    if st.button("Clear conversation memory"):
        try:
            call_api("/reset", {"session_id": st.session_state.session_id})
        except requests.exceptions.RequestException:
            pass
        st.session_state.history = []
        st.rerun()
