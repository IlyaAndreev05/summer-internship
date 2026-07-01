"""Rich-based console UI for the ALINA GPSS Consultant."""

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from alina_rag.config import settings
from alina_rag.domain.models import AgentStep, BotPlatform, UserId

if TYPE_CHECKING:
    from alina_rag.application.chat_service import ChatService

logger = logging.getLogger(__name__)


def _on_step(console: Console, step: AgentStep) -> None:
    """Print one ReAct step to the console (verbose mode only)."""
    if step.thought:
        console.print(Panel(
            step.thought,
            title="[bold blue]Thought[/]",
            border_style="blue",
        ))
    if step.action_input:
        label = "search_documents" if "search" in step.action else "search_keywords"
        console.print(Text(f"  🔍 {label}(\"{step.action_input}\")", style="dim yellow"))
    if step.observation:
        obs = step.observation[:300] + "..." if len(step.observation) > 300 else step.observation
        console.print(Text(f"  📄 {obs}", style="dim green"))


async def run_console(chat_service: "ChatService") -> None:
    """Run the interactive console UI."""
    console = Console()
    user_id = UserId(BotPlatform.CONSOLE, "anonymous")

    console.print()
    console.print("[bold cyan]╔════════════════════════════════════════════╗[/]")
    console.print("[bold cyan]║[/]    [bold yellow]ALINA GPSS AI Консультант[/]   [bold cyan]║[/]")
    console.print("[bold cyan]╚════════════════════════════════════════════╝[/]")
    console.print("[dim]/exit — выход    /clear — очистить историю    /verbose — показать мысли агента[/]")
    console.print()

    verbose = settings.chat_verbose

    while True:
        try:
            user_input = await asyncio.to_thread(Prompt.ask, "[bold green]Вы[/]")
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
            console.print("[dim]История очищена.[/]")
            continue

        if stripped == "/verbose":
            verbose = not verbose
            state = "включён" if verbose else "выключен"
            console.print(f"[dim]Verbose-режим {state}.[/]")
            continue

        # Build callback only if verbose
        step_cb = (lambda s: _on_step(console, s)) if verbose else None

        # Hide spinner in verbose mode (steps are the progress indicator)
        cm = console.status("[dim]Думаю...[/]", spinner="dots") if not verbose else contextlib.nullcontext()
        with cm:
            try:
                response = await chat_service.handle_message(
                    user_id, stripped, step_callback=step_cb,
                )
            except Exception:
                logger.exception("Chat service error")
                console.print("[red]Ошибка при обработке запроса.[/]")
                continue

        console.print()
        console.print(Markdown(response))
        console.print()
