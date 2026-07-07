import sys
from pathlib import Path

from ..config import Settings
from ..rag_agent import RAGAgent


class BatchMode:
    def __init__(self, agent: RAGAgent, settings: Settings):
        self.agent = agent
        self.settings = settings

    def run(self):
        input_dir = Path(self.settings.batch_input_dir)
        output_dir = Path(self.settings.batch_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob("*"))
        text_files = [f for f in files if f.suffix.lower() in (".txt", ".md")]

        if not text_files:
            print("Ошибка: нет входных файлов в data/batch/input")
            sys.exit(1)

        for f in text_files:
            query = f.read_text(encoding="utf-8").strip()
            answer = self.agent.answer(query)
            out_path = output_dir / f"{f.stem}_answer.txt"
            out_path.write_text(answer, encoding="utf-8")
            print(f"Обработан {f.name} -> {out_path.name}")
