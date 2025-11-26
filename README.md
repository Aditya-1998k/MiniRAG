# MiniRAG

MiniRAG is a lightweight, modular **Retrieval-Augmented Generation (RAG)** system designed to run **fully locally** — even on low-resource machines like the MacBook Air M1.  
It demonstrates the complete RAG pipeline end-to-end using simple components:

- **SentenceTransformers** for text embeddings  
- **FAISS** for vector search  
- **Qwen 0.5B** for fast, CPU-friendly LLM inference  
- Clean, modular architecture for easy learning and extension  

MiniRAG is perfect for anyone who wants to understand how RAG works internally without heavy frameworks or cloud dependencies.

1. Using **Ollama** for clean architecture. It will help in running the model locally and we can use
api endpoint for generator.

2. Using **pytorch** with HF transformer to run model locally

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
├── core/
│ ├── __init__.py
│ ├── config.py
│ ├── embedder.py
│ ├── retriever.py
│ ├── generator.py
|
│ ├── pytorch_llm/
│ ├── __init__.py
│ ├── llm.py
|
│ ├── ollama/
│ ├── __init__.py
│ ├── ollama.py
│
├── embeddings/
│ ├── index.faiss
│ ├── chunks.pkl
│
├── data/
│ ├── story.txt
│
├── ingest.py
├── main.py
├── .env
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```
And set .env
```
MINIRAG_BACKEND=ollama  #pytorch for using HF transformer
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

## Running Model with Ollama locally
1. Install ollama
```
brew install ollama -macos
sudo apt install ollama - linux
```

2. Start ollama service
```
ollama serve
```

3. Pull the model locally
```
ollama pull qwen2.5:0.5b
```
4. Run the model
```
ollama run qwen2.5:0.5b
```
5.  Try with simple question
```
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"
```
6. Python integration
```python
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"
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



