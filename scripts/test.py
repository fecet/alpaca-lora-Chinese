from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "decapoda-research/llama-13b-hf",
    cache_dir="data/hf",
    load_in_8bit=True,
    device_map="auto",
)
