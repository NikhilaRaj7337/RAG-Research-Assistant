"""
Central place to pick the LLM + embeddings provider based on LLM_PROVIDER in .env.

Supports:
  - "gemini" (default): Google's free-tier API. Get a key at
    https://aistudio.google.com/apikey — no credit card required.
  - "openai": paid, requires OPENAI_API_KEY with billing set up.
"""
import os

def get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").lower()


def get_chat_model(temperature: float = 0.0):
    provider = get_provider()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
        return ChatOpenAI(model=model, temperature=temperature)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'gemini' or 'openai'.")


def get_embedder():
    provider = get_provider()
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        return GoogleGenerativeAIEmbeddings(model=model)
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'gemini' or 'openai'.")
