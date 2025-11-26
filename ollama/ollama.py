
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"

def load_ollama(model_name: str = MODEL_NAME):
    """
    Nothing to load like PyTorch models.
    Just return the model name so other modules can use it.
    """
    return model_name


def generate_with_ollama(model_name: str, question: str, context: str) -> str:
    """
    Send a RAG prompt to Ollama using its REST API.
    """

    prompt = f"""Use ONLY the provided context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,   # set True for streaming
    }

    response = requests.post(OLLAMA_URL, json=payload)
    data = response.json()

    return data.get("response", "").strip()
