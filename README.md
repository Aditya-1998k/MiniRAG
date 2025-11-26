# MiniRAG

MiniRAG is a lightweight, modular **Retrieval-Augmented Generation (RAG)** system designed to run **fully locally** — even on low-resource machines like the MacBook Air M1.  
It demonstrates the complete RAG pipeline end-to-end using simple components:

- **SentenceTransformers** for text embeddings  
- **FAISS** for vector search  
- **Qwen 0.5B** for fast, CPU-friendly LLM inference  
- Clean, modular architecture for easy learning and extension  

MiniRAG is perfect for anyone who wants to understand how RAG works internally without heavy frameworks or cloud dependencies.

---

## ✨ Features

- 🔍 **Local Embedding Search** using FAISS  
- 🧩 **Chunking + Embedding + Indexing** pipeline  
- ⚡ **Fast CPU LLM** with Qwen 0.5B  
- 📦 **Modular File Structure** (retriever, generator, embedder, config)  
- 🛠️ **Runs on Mac, Linux, Windows**  
- 💡 Minimal, readable code ideal for learning  

---

## 📁 Project Structure

```
MiniRAG/
│
├── rag/
│ ├── __init__.py
│ ├── config.py # environment and performance settings
│ ├── llm.py # loads the local LLM
│ ├── embedder.py # embedding model
│ ├── retriever.py # FAISS retrieval logic
│ ├── generator.py # answer generation logic
│
├── embeddings/
│ ├── index.faiss # stored FAISS index
│ ├── chunks.pkl # text chunks mapped to vectors
│
├── data/
│ ├── story.txt
│
├── ingest.py # builds FAISS index from text chunks
├── rag_local.py # main script to query the RAG
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Ingest data and build the FAISS index
```bash
python3 ingest.py
```

### 3. Run the local RAG system
```bash
python3 rag_local.py
```

### 4. Ask a question
```bash
What were the three locations marked on Eldon Marr’s original map?
```

## ⚙️ How It Works

MiniRAG follows a simple, transparent pipeline:
1. **Chunking** – split long documents into small text chunks
2. **Encoding** – convert chunks into embeddings using all-MiniLM-L6-v2
3. **Indexing** – store vectors in a FAISS index for fast retrieval
4. **Retrieval** – find the top-k relevant chunks for a user query
5. **Generation** – feed context + question to a small local LLM

**Return** – produce a grounded answer using retrieved information

## 🧠 Models Used
1. **Embeddings**
- model name: `sentence-transformers/all-MiniLM-L6-v2`
- Local LLM (fast on CPU)
- `Qwen/Qwen2.5-0.5B-Instruct`
- ~1.8 GB RAM usage
- 5–10× faster than bigger models
- Very good accuracy for RAG

## 🛠️ Requirements
- Python 3.10+
- 4 GB+ free RAM (MacBook Air M1 supported)
- No GPU required



