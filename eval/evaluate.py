"""
Evaluate the RAG assistant against a labeled test set.

Test set format (eval/test_set.jsonl), one JSON object per line:
  {
    "question": "...",
    "reference_answer": "...",
    "gold_source": "data/sample_docs/some_file.txt"   # substring match against retrieved metadata
  }

Reports, averaged over the test set:
  - answer_relevance   : cosine similarity between answer & reference embeddings (0-1)
  - retrieval_precision_at_k : fraction of questions where gold_source appears in
                                the top-k retrieved chunks
  - hallucination_rate : fraction of answer sentences NOT supported by retrieved
                          context, as judged by an LLM grader (0 = fully grounded)

Run this BEFORE and AFTER a change (e.g. switching chunking strategy) to get the
kind of before/after delta you'd want to cite as an actual, measured number.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
load_dotenv()

HALLUCINATION_GRADER_PROMPT = """You are grading whether an AI assistant's answer is \
fully supported by the given context. Break the ANSWER into individual factual claims. \
For each claim, decide if it is directly supported by the CONTEXT (yes/no). \
Return ONLY a JSON object: {{"total_claims": N, "unsupported_claims": M}}.

CONTEXT:
{context}

ANSWER:
{answer}
"""


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def grade_hallucination(llm, answer: str, context: str) -> tuple[int, int]:
    resp = llm.invoke(HALLUCINATION_GRADER_PROMPT.format(context=context, answer=answer))
    text = resp.content.strip()
    # Some models wrap JSON in markdown code fences even when told not to.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
        return int(data["total_claims"]), int(data["unsupported_claims"])
    except (json.JSONDecodeError, KeyError, ValueError):
        print(f"    [grader parse failed] raw response: {text!r}")
        return 0, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--test_set", required=True)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    from chains import RagSession
    from providers import get_chat_model, get_embedder

    session = RagSession(args.index_dir, k=args.k)
    embedder = get_embedder()
    grader_llm = get_chat_model(temperature=0.0)

    with open(args.test_set) as f:
        test_cases = [json.loads(line) for line in f if line.strip()]

    relevances, precisions, total_claims_sum, unsupported_sum = [], [], 0, 0

    for case in test_cases:
        session.reset()
        result = session.ask(case["question"])
        answer = result["answer"]
        sources = result["source_documents"]

        # 1. Answer relevance vs. reference
        ans_vec = embedder.embed_query(answer)
        ref_vec = embedder.embed_query(case["reference_answer"])
        relevances.append(cosine_sim(ans_vec, ref_vec))

        # 2. Retrieval precision@k
        gold = case.get("gold_source")
        hit = any(gold and gold in doc.metadata.get("source", "") for doc in sources)
        precisions.append(1.0 if hit else 0.0)

        # 3. Hallucination proxy
        context = "\n\n".join(doc.page_content for doc in sources)
        total, unsupported = grade_hallucination(grader_llm, answer, context)
        total_claims_sum += total
        unsupported_sum += unsupported

        time.sleep(60)

    n = len(test_cases)
    print(f"Evaluated {n} questions\n")
    print(f"Answer relevance (mean cosine sim): {np.mean(relevances):.3f}")
    print(f"Retrieval precision@{args.k}:         {np.mean(precisions):.3f}")
    if total_claims_sum > 0:
        print(f"Hallucination rate (unsupported/total claims): {unsupported_sum / total_claims_sum:.3f}")
    else:
        print("Hallucination rate: N/A (grader returned no parseable claims — check grader prompt/model)")


if __name__ == "__main__":
    main()
