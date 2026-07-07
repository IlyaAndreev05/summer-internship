import sys
import time
from ..indexer import Indexer
from ..rag_agent import RAGAgent


class ConsoleMode:
    def __init__(self, agent: RAGAgent, indexer: Indexer):
        self.agent = agent
        self.indexer = indexer

    def run(self):
        print("╔════════════════════════════════════════════════╗")
        print("║        GPSS AI Консультант                     ║")
        print("╚════════════════════════════════════════════════╝")
        print("/exit — выход    /clear — очистить историю")
        while True:
            try:
                query = input("\nВы: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query == "/exit":
                break
            if query == "/clear":
                self.agent.clear_history()
                print("История очищена")
                continue
            if not query.strip():
                continue
            if not self.indexer.indexed:
                done, total, phase, start_time = self.indexer.progress
                elapsed = time.time() - start_time if start_time else 0
                phase_label = {"loading": "загрузка файлов", "embedding": "эмбеддинги", "bm25": "построение BM25"}.get(phase, phase)
                if total > 0:
                    print(f"Индексация [{phase_label}]: {done}/{total}, прошло {elapsed:.0f}с")
                else:
                    print(f"Индексация [{phase_label}]... прошло {elapsed:.0f}с")
                if self.indexer.error:
                    print(f"Ошибка индексации: {self.indexer.error}")
                    sys.exit(1)
                continue
            answer = self.agent.answer(query)
            print(f"Ассистент: {answer}")
