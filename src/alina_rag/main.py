import logging

import typer

from alina_rag.agent import RAGAgent
from alina_rag.batch_mode import run_batch
from alina_rag.config import settings
from alina_rag.console_bot import run_console
from alina_rag.indexer import index_documents
from alina_rag.test_mode import run_tests
from alina_rag.vk_bot import run_vk_bot

app = typer.Typer(name="alina-rag", help="ALINA GPSS AI Consultant")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@app.command()
def console(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show search debug info"),
) -> None:
    agent = RAGAgent()
    run_console(agent, verbose=verbose)


@app.command()
def vk() -> None:
    if not settings.vk_token or not settings.vk_group_id:
        typer.echo("Error: VK_TOKEN and VK_GROUP_ID must be set in .env")
        raise typer.Exit(code=1)
    agent = RAGAgent()
    run_vk_bot(agent, token=settings.vk_token, group_id=settings.vk_group_id)


@app.command()
def index() -> None:
    agent = RAGAgent()
    count = index_documents(agent)
    typer.echo(f"Indexed {count} chunks")


@app.command()
def batch() -> None:
    agent = RAGAgent()
    run_batch(agent)
    typer.echo("Batch processing complete")


@app.command()
def test() -> None:
    agent = RAGAgent()
    run_tests(agent)


if __name__ == "__main__":
    app()
