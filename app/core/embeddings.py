from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from typing import List
import numpy as np

# all-mpnet-base-v2 is a strong general-purpose embedding model,
# good balance of quality vs speed, runs comfortably on your 8GB GPU
_model = SentenceTransformer("all-mpnet-base-v2")


def embed_text(text: str) -> np.ndarray:
    """Turn a single piece of text into an embedding vector."""
    return _model.encode(text, convert_to_numpy=True)


def embed_batch(texts: List[str]) -> np.ndarray:
    """
    Embed multiple texts at once - much faster than calling embed_text()
    in a loop, since the model can batch the computation.
    """
    return _model.encode(texts, convert_to_numpy=True)


def similarity(text_a: str, text_b: str) -> float:
    """
    Returns a similarity score between 0 and 1 (roughly) for two pieces
    of text. 1 = near-identical meaning, 0 = unrelated.
    """
    emb_a = embed_text(text_a)
    emb_b = embed_text(text_b)
    return float(cos_sim(emb_a, emb_b)[0][0])


if __name__ == "__main__":
    # quick sanity test
    a = "Built end-to-end deep learning pipelines using PyTorch"
    b = "Experience developing production machine learning systems"
    c = "Managed a team of sales representatives"

    print(f"Similar sentences score: {similarity(a, b):.3f}")
    print(f"Unrelated sentences score: {similarity(a, c):.3f}")