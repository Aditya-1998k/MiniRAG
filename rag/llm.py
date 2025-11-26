
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="float32",
        low_cpu_mem_usage=True
    ).to("cpu")

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_new_tokens=100,
        do_sample=False,
        temperature=0.0
    )

    return generator
