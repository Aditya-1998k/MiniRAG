import os
from core.config import setup_environment
from pytorch_llm.llm import load_llm
from core.embedder import load_embedder
from core.retriever import load_retrieval_index, retrieve


os.environ["TRANSFORMERS_VERBOSITY"] = "error"
BACKEND = os.environ.get("MINIRAG_BACKEND", "pytorch")

if BACKEND == "pytorch":
    from pytorch_llm.llm import load_llm
    from pytorch_llm.generator import generate_answer
elif BACKEND == "ollama":
    from ollama.ollama import load_ollama as load_llm
    from ollama.ollama import generate_with_ollama as generate_answer
else:
    raise ValueError(f"Unknown backend: {BACKEND}")



if __name__ == "__main__":
    try:
        breakpoint()
        setup_environment()

        model = load_llm()
        embedder = load_embedder()
        index, chunks = load_retrieval_index()

        question = input("\nAsk a question: ")

        context_chunks = retrieve(question, embedder, index, chunks)
        context_text = "\n\n".join(context_chunks)

        answer = generate_answer(model, question, context_text)

        print("\n--- ANSWER ---\n")
        print(answer)

    except Exception as e:
        print("ERROR:", e)
