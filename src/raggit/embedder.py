from typing import Callable, List


class Embedder:
    def __init__(self, model_name: str, embed_fn: Callable[[str], List[float]]):
        self.model_name = model_name
        self.embed_fn = embed_fn

    def embed(self, text: str) -> List[float]:
        return self.embed_fn(text)
