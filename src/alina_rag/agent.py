import logging
from collections.abc import Callable

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from alina_rag.config import settings
from alina_rag.search import BM25Search, HybridSearch, ProjectSearch, TrigramSearch, VectorSearch

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """Ты консультант по GPSS. Ответь на вопрос пользователя, используя ТОЛЬКО информацию из результатов поиска ниже. Если информации недостаточно или её нет — скажи «В документации не найдено информации». Отвечай кратко, на русском, без нумерации.

Результаты поиска по документации:
{docs_results}

Проекты:
{projects_results}

Вопрос: {question}"""


class RAGAgent:
    def __init__(self):
        self._llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_host,
            temperature=0.1,
        )
        qdrant = QdrantClient(url=settings.qdrant_url)
        embeddings = OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_host,
        )
        try:
            qdrant.get_collection(settings.qdrant_collection)
        except Exception:
            qdrant.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
        self._docs = HybridSearch([
            VectorSearch(qdrant, settings.qdrant_collection, embeddings),
            TrigramSearch(),
            BM25Search(),
        ])
        self._projects = ProjectSearch()

    def get_llm(self):
        return self._llm

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        step_callback: Callable | None = None,
    ) -> str:
        docs_result = self._docs.search(question)
        projects_result = self._projects.search(question)

        if step_callback:
            step_callback({"type": "tool_call", "tool": "search_docs", "query": question, "iteration": 0})
            step_callback({"type": "tool_call", "tool": "search_projects", "query": question, "iteration": 0})

        prompt = ANSWER_PROMPT.format(
            docs_results=docs_result,
            projects_results=projects_result,
            question=question,
        )

        messages: list = []
        if history:
            for msg in history[-6:]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self._llm.invoke(messages)
            return response.content if isinstance(response.content, str) else str(response.content)
        except Exception:
            logger.exception("LLM call failed")
            return "Ошибка при генерации ответа."
