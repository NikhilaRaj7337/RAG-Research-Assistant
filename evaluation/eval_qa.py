"""
Evaluation harness for the RAG assistant.

Produces the two numbers you'd cite on a resume:
  - answer relevance %  : does the answer actually address the question,
                           judged against a reference answer (LLM-as-judge)
  - grounded rate %     : is the answer supported by the retrieved context
                           (i.e. NOT hallucinated), also LLM-as-judge

This gives you real, reproducible numbers instead of made-up ones -- rerun
it after any chunking/retrieval change and the README table below will
reflect what your system actually does.

Usage:
    python -m evaluation.eval_qa
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import CONFIG
from src.rag_chain import ResearchAssistant

JUDGE_MODEL = "gemini-3.6-flash"

RELEVANCE_PROMPT = ChatPromptTemplate.from_template(
    "Question: {question}\n"
    "Reference answer: {reference}\n"
    "System answer: {answer}\n\n"
    "On a scale of 1-5, how relevant is the system answer to the question, "
    "compared to what the reference answer covers? "
    "Reply with ONLY the integer."
)

GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_template(
    "Context:\n{context}\n\n"
    "Answer: {answer}\n\n"
    "Is every factual claim in the answer supported by the context above? "
    "Reply with ONLY 'yes' or 'no'. Reply 'no' if the answer adds information "
    "not present in the context, or if the answer says it doesn't have enough "
    "information (that case counts as ungrounded/skipped, not a hallucination "
    "-- reply 'skip' in that case instead)."
)


def run_eval():
    test_set = json.loads(Path(__file__).parent.joinpath("test_questions.json").read_text())
    assistant = ResearchAssistant(CONFIG)
    judge = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0)

    relevance_scores = []
    grounded_flags = []
    rows = []

    for item in test_set:
        q, ref = item["question"], item["reference_answer"]
        result = assistant.ask(q, session_id="eval")
        answer = result["answer"]

        # retrieve raw context text for groundedness check
        retriever = assistant._get_retriever()
        docs = retriever.invoke(q)
        context_text = "\n\n".join(d.page_content for d in docs)

        rel_raw = judge.invoke(
            RELEVANCE_PROMPT.format(question=q, reference=ref, answer=answer)
        ).content.strip()
        try:
            rel_score = int("".join(c for c in rel_raw if c.isdigit()) or 0)
        except ValueError:
            rel_score = 0
        relevance_scores.append(rel_score)

        ground_raw = judge.invoke(
            GROUNDEDNESS_PROMPT.format(context=context_text, answer=answer)
        ).content.strip().lower()
        if "yes" in ground_raw:
            grounded_flags.append(1)
        elif "no" in ground_raw:
            grounded_flags.append(0)
        # "skip" (declined to answer) excluded from the hallucination denominator

        rows.append({"question": q, "answer": answer, "relevance": rel_score, "grounded": ground_raw})
        assistant.reset("eval")

    avg_relevance_pct = (sum(relevance_scores) / (len(relevance_scores) * 5)) * 100
    hallucination_rate_pct = (
        100 * (1 - sum(grounded_flags) / len(grounded_flags)) if grounded_flags else 0.0
    )

    print("\n=== Evaluation Results ===")
    for r in rows:
        print(f"- [{r['relevance']}/5, grounded={r['grounded']}] {r['question']}")
    print(f"\nAverage answer relevance: {avg_relevance_pct:.1f}%")
    print(f"Hallucination rate:       {hallucination_rate_pct:.1f}%")
    print("\nUse these numbers (not guessed ones) in your README / resume.")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "last_run.json").write_text(json.dumps({
        "rows": rows,
        "avg_relevance_pct": avg_relevance_pct,
        "hallucination_rate_pct": hallucination_rate_pct,
    }, indent=2))


if __name__ == "__main__":
    run_eval()
