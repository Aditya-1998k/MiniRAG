
from sentence_transformers import SentenceTransformer

def load_embedder():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
