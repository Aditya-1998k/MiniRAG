import os
from rag.config import setup_environment
from rag.llm import load_llm
from rag.embedder import load_embedder
from rag.retriever import load_retrieval_index, retrieve
from rag.generator import generate_answer


os.environ["TRANSFORMERS_VERBOSITY"] = "error"


if __name__ == "__main__":
    try:
        breakpoint()
        setup_environment()

        generator = load_llm()
        embedder = load_embedder()
        index, chunks = load_retrieval_index()

        question = input("\nAsk a question: ")

        context_chunks = retrieve(question, embedder, index, chunks)
        context_text = "\n\n".join(context_chunks)

        answer = generate_answer(generator, question, context_text)

        print("\n--- ANSWER ---\n")
        print(answer)

    except Exception as e:
        print("ERROR:", e)
