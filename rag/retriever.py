
import faiss
import pickle

def load_retrieval_index(index_path="embeddings/index.faiss",
                         chunks_path="embeddings/chunks.pkl"):
    index = faiss.read_index(index_path)
    chunks = pickle.load(open(chunks_path, "rb"))
    return index, chunks


def retrieve(query, embedder, index, chunks, top_k=3):
    query_vec = embedder.encode([query])
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i >= 0]
