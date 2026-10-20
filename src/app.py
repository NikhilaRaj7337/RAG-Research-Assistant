"""CLI chat loop for the RAG research assistant."""
import argparse

from chains import RagSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve")
    args = parser.parse_args()

    session = RagSession(args.index_dir, k=args.k)
    print("RAG Research Assistant — type 'exit' to quit, 'reset' to clear memory.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "reset":
            session.reset()
            print("(conversation memory cleared)\n")
            continue

        result = session.ask(question)
        print(f"\nAssistant: {result['answer']}\n")

        if result["source_documents"]:
            print("Sources:")
            for doc in result["source_documents"]:
                src = doc.metadata.get("source", "unknown")
                print(f"  - {src}")
            print()


if __name__ == "__main__":
    main()
