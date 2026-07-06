from __future__ import annotations

import logging

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text

from alina_rag.agent import RAGAgent
from alina_rag.indexer import Indexer
from alina_rag.models import ChatHistory

logger = logging.getLogger(__name__)


def run_console(agent: RAGAgent, indexer: Indexer, verbose: bool = False) -> None:
    console = Console()
    history = ChatHistory()

    console.print()
    console.print("[bold cyan]╔════════════════════════════════════════════════╗[/]")
    console.print("[bold cyan]║[/]    [bold yellow]ALINA GPSS AI Консультант[/]              [bold cyan]║[/]")
    console.print("[bold cyan]╚════════════════════════════════════════════════╝[/]")
    console.print("[dim]/exit — выход    /clear — очистить историю    /verbose — показать поиск[/]")
    console.print()

    while True:
        try:
            user_input = Prompt.ask("[bold green]Вы[/]")
        except (KeyboardInterrupt, EOFError):
            console.print()
            console.print("[dim]До свидания![/]")
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        if stripped == "/exit":
            console.print("[dim]До свидания![/]")
            break

        if stripped == "/clear":
            history.clear()
            console.print("[dim]История очищена.[/]")
            continue

        if stripped == "/verbose":
            verbose = not verbose
            state = "включён" if verbose else "выключен"
            console.print(f"[dim]Режим отладки {state}.[/]")
            continue

        if not indexer.is_ready:
            console.print("[yellow]⏳ Индексация в процессе, подождите...[/]")
            continue

        def step_cb(step: dict, _verbose: bool = verbose) -> None:
            if not _verbose:
                return
            if step.get("type") == "tool_call":
                tool = step.get("tool", "?")
                query = step.get("query", "")
                iteration = step.get("iteration", 0) + 1
                console.print(
                    Text(
                        f'  🔍 {tool}("{query}") [итерация {iteration}]',
                        style="dim yellow",
                    )
                )
            elif step.get("type") == "search":
                query = step.get("query", "")
                results = step.get("results")
                if results:
                    count = len(results)
                    top_src = results[0].source if results else "?"
                    top_preview = results[0].text[:80] if results and results[0].text else ""
                    console.print(
                        Text(
                            f'  🔍 поиск("{query}") — найдено {count}, топ: {top_src}\n'
                            f'     {top_preview}…',
                            style="dim yellow",
                        )
                    )
                else:
                    console.print(
                        Text(f'  🔍 поиск("{query}")', style="dim yellow")
                    )

        try:
            response = agent.answer(
                stripped,
                history=history.last_dicts(10),
                step_callback=step_cb,
            )
        except Exception:
            logger.exception("Agent error")
            console.print("[red]Ошибка при обработке запроса.[/]")
            continue

        history.add_user(stripped)
        history.add_assistant(response)

        console.print()
        console.print(Markdown(response))
        console.print()
