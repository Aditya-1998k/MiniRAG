
def generate_answer(generator, question, context):
    prompt = f"""You are a helpful assistant. Use ONLY the provided context to answer the question.
Do NOT include any extra instructions or meta text.

Context:
{context}

Question:
{question}

Answer:
"""
    output = generator(prompt)[0]["generated_text"]
    breakpoint()
    return output
