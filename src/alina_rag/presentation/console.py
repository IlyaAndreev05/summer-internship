"""Rich-based console UI for the ALINA GPSS Consultant."""

import asyncio
import logging
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from alina_rag.domain.models import BotPlatform, UserId

if TYPE_CHECKING:
    from alina_rag.application.chat_service import ChatService

logger = logging.getLogger(__name__)


async def run_console(chat_service: "ChatService") -> None:
    """Run the interactive console UI.

    Handles /exit and /clear commands locally.
    Ctrl+C triggers graceful exit.
    """
    console = Console()
    user_id = UserId(BotPlatform.CONSOLE, "anonymous")

    console.print()
    console.print(
        "[bold cyan]╔════════════════════════════════════════════╗[/]"
    )
    console.print(
        "[bold cyan]║[/]    [bold yellow]ALINA GPSS AI Консультант[/]   [bold cyan]║[/]"
    )
    console.print(
        "[bold cyan]╚════════════════════════════════════════════╝[/]"
    )
    console.print("[dim]/exit — выход    /clear — очистить историю[/]")
    console.print()

    while True:
        try:
            user_input = await asyncio.to_thread(
                Prompt.ask, "[bold green]Вы[/]"
            )
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

        with console.status("[dim]Думаю...[/]", spinner="dots"):
            try:
                response = await chat_service.handle_message(
                    user_id, stripped
                )
            except Exception:
                logger.exception("Chat service error")
                console.print("[red]Ошибка при обработке запроса.[/]")
                continue

        console.print()
        console.print(Markdown(response))
        console.print()
