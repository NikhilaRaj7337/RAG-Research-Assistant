"""
Terminal chat interface for the RAG research assistant.

Usage:
    python -m src.ingest      # once, to build the index
    python cli.py             # start chatting
"""
from rich.console import Console
from rich.markdown import Markdown

from src.config import CONFIG
from src.rag_chain import ResearchAssistant

console = Console()


def main():
    console.print("[bold cyan]LLM Research Assistant (RAG)[/bold cyan]")
    console.print("Type your question, or 'reset' to clear memory, 'exit' to quit.\n")

    assistant = ResearchAssistant(CONFIG)
    session_id = "cli-session"

    while True:
        question = console.input("[bold green]You:[/bold green] ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "reset":
            assistant.reset(session_id)
            console.print("[yellow]Conversation memory cleared.[/yellow]\n")
            continue

        result = assistant.ask(question, session_id=session_id)
        console.print("\n[bold blue]Assistant:[/bold blue]")
        console.print(Markdown(result["answer"]))
        if result["sources"]:
            console.print(f"[dim]Sources: {', '.join(result['sources'])}[/dim]\n")
        else:
            console.print()


if __name__ == "__main__":
    main()
