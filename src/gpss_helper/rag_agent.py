import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .bm25_search import BM25Search
from .config import Settings
from .models import SearchResult
from .vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — ИИ-консультант по системе GPSS (General Purpose Simulation System).
Отвечай на вопросы пользователя, используя ТОЛЬКО предоставленный контекст.
Если в контексте нет ответа, скажи об этом честно.
Отвечай на русском языке.

Контекст:
{context}"""


class RAGAgent:
    def __init__(
        self, settings: Settings, vector_store: QdrantVectorStore, bm25: BM25Search
    ):
        self.settings = settings
        self.vector_store = vector_store
        self.bm25 = bm25
        self.llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_url,
            temperature=0.1,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        self._history: list[dict] = []

    def _fuse_results(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        for r in vector_results:
            key = r.content[:200]
            scores[key] = r.score
            result_map[key] = r

        for r in bm25_results:
            key = r.content[:200]
            scores[key] = scores.get(key, 0) + r.score
            if key not in result_map:
                result_map[key] = r

        sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [result_map[k] for k, _ in sorted_keys]

    def search(self, query: str) -> list[SearchResult]:
        vector_results = self.vector_store.search(
            query, limit=self.settings.top_k * 2
        )
        bm25_results = self.bm25.search(query, limit=self.settings.top_k * 2)
        return self._fuse_results(vector_results, bm25_results, self.settings.top_k)

    def answer(self, question: str) -> str:
        results = self.search(question)
        if not results:
            return "Не удалось найти релевантную информацию по вашему вопросу."

        context_parts: list[str] = []
        for r in results:
            source_info = f"[{r.source}]"
            if r.project_meta:
                source_info = f"[Проект: {r.project_meta.name}]"
            context_parts.append(f"{source_info}\n{r.content}")

        context = "\n\n---\n\n".join(context_parts)
        response = self.chain.invoke({"context": context, "question": question})

        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self) -> None:
        self._history.clear()
