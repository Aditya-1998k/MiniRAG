# This file is helper for conversion
# text --> Chunks --> embeddings --> FAISS Index


from sentence_transformers import SentenceTransformer
import faiss
import pickle

def load_text(path):
    return open(path, "r", encoding="utf-8").read()

def chunk_text(text, chunk_size=250):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])  # FIX: convert list → string
        chunks.append(chunk)
    return chunks

def build_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    vectors = model.encode(
        chunks,
        show_progress_bar=True
    )

    dimension = vectors.shape[1]  # FIX: correct dimension
    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    return index

if __name__ == "__main__":
    text = load_text("data/story.txt")
    chunks = chunk_text(text)

    index = build_index(chunks)

    faiss.write_index(index, "embeddings/index.faiss")
    pickle.dump(chunks, open("embeddings/chunks.pkl", "wb"))

    print("Ingestion Complete!")
