from transformers import LlamaForCausalLM, GenerationConfig
from peft import PeftModel
import torch


def generate_instruction_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


def evaluate(
    model,
    tokenizer,
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
    max_token=256,
):
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        top_k=40,
        no_repeat_ngram_size=3,
    )
    prompt = generate_instruction_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_token,
    )
    s=generation_output.sequences[0]
    output = tokenizer.decode(s)
    res = output.split("### Response:")[1].strip()
    print("Response:", res)
    return res


def load_lora(lora_path, base_model="decapoda-research/llama-7b-hf"):
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto",
        cache_dir="data/hf",
    )
    lora = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16)
    return lora
