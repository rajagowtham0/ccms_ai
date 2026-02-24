import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not isinstance(texts, list):
            raise TypeError("Input must be a list of strings.")

        if len(texts) == 0:
            raise ValueError("Text list is empty.")

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

    def generate_single_embedding(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        return self.model.encode(
            text,
            convert_to_numpy=True
        )