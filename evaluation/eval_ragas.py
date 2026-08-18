"""
RAGAS evaluation harness.

This is the industry-standard eval library for RAG systems, measuring four
distinct failure modes separately instead of one fuzzy "relevance" score:

  - faithfulness       : is the answer actually supported by the retrieved
                          context (the hallucination check)
  - answer_relevancy    : does the answer address what was asked
  - context_precision   : of the chunks retrieved, how many were relevant
  - context_recall      : of the relevant info that exists, how much did
                          retrieval actually surface

context_precision/recall are what make this stronger than the custom
eval_qa.py harness -- they isolate retrieval quality from generation
quality, so you can tell whether a bad answer came from bad retrieval or
the LLM ignoring good context.

Usage:
    python -m evaluation.eval_ragas
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import CONFIG
from src.rag_chain import ResearchAssistant


def run_eval():
    test_set = json.loads(Path(__file__).parent.joinpath("test_questions.json").read_text())
    assistant = ResearchAssistant(CONFIG)

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in test_set:
        q, ref = item["question"], item["reference_answer"]
        result = assistant.ask(q, session_id="ragas-eval")

        docs = assistant.retriever.invoke(q)
        contexts = [d.page_content for d in docs]

        rows["question"].append(q)
        rows["answer"].append(result["answer"])
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(ref)
        assistant.reset("ragas-eval")

    dataset = Dataset.from_dict(rows)

    # RAGAS needs an LLM + embeddings to act as the judge; reuse Gemini so
    # this doesn't require a second API key.
    judge_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=CONFIG.chat_model, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=CONFIG.embedding_model)
    )

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    df = results.to_pandas()
    print("\n=== RAGAS Evaluation Results ===")
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])

    summary = {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "context_precision": float(df["context_precision"].mean()),
        "context_recall": float(df["context_recall"].mean()),
    }
    print("\nAverages:")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "ragas_last_run.csv", index=False)
    (out_dir / "ragas_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved detailed results to {out_dir}/ragas_last_run.csv")


if __name__ == "__main__":
    run_eval()
