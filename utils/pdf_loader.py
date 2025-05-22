import os
from typing import List
from PyPDF2 import PdfReader
from loguru import logger


class PDFLoader:
    def __init__(self):
        pass

    def load(self, filepath: str) -> List[str]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            return self._load_pdf(filepath)
        elif ext in [".txt", ".md"]:
            return self._load_text(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _load_pdf(self, filepath: str) -> List[str]:
        try:
            reader = PdfReader(filepath)
            texts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
            logger.info(f"Loaded {len(texts)} pages from PDF {filepath}")
            return texts
        except Exception as e:
            logger.error(f"Failed to load PDF {filepath}: {e}")
            raise e

    def _load_text(self, filepath: str) -> List[str]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            logger.info(f"Loaded {len(lines)} lines from text file {filepath}")
            return lines
        except Exception as e:
            logger.error(f"Failed to load text file {filepath}: {e}")
            raise e
