
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    model = model.to("cpu")

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_new_tokens=100,
        do_sample=False,
    )

    return generator
