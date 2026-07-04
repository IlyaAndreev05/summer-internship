import logging

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text

from alina_rag.agent import RAGAgent

logger = logging.getLogger(__name__)


def run_console(agent: RAGAgent, verbose: bool = False) -> None:
    """Запуск интерактивной консоли для общения с агентом."""
    console = Console()
    history: list[dict] = []

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

        def step_cb(step, _verbose: bool = verbose):
            """Callback для вывода информации о поиске в verbose-режиме."""
            if not _verbose:
                return
            if step.get("type") == "tool_call":
                console.print(
                    Text(
                        f"  🔍 {step.get('tool', '?')}(\"{step.get('query', '')}\") [итерация {step.get('iteration', 0)+1}]",
                        style="dim yellow",
                    )
                )

        try:
            response = agent.answer(
                stripped, history=history[-10:] if history else None, step_callback=step_cb
            )
        except Exception:
            logger.exception("Agent error")
            console.print("[red]Ошибка при обработке запроса.[/]")
            continue

        history.append({"role": "user", "content": stripped})
        history.append({"role": "assistant", "content": response})

        console.print()
        console.print(Markdown(response))
        console.print()
