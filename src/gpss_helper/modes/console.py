import sys

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
                print("Идёт процесс индексации...")
                if self.indexer.error:
                    print(f"Ошибка индексации: {self.indexer.error}")
                    sys.exit(1)
                continue
            answer = self.agent.answer(query)
            print(f"Ассистент: {answer}")
