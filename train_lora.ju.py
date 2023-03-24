# %%

import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CONDA_PREFIX"] = "/home/rok/.conda/envs/diff"

import bitsandbytes as bnb

# del os.environ["CUDA_PATH"]
# %%

import torch
import torch.nn as nn
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

# from peft import LoraConfig, get_peft_model
torch.backends.cuda.matmul.allow_tf32 = True

# %%
MICRO_BATCH_SIZE = 128  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 384
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3  # we don't need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size


# %%
def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
{data_point["output"]}"""


def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


# %%

model = AutoModelForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map=device_map,
)
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "decapoda-research/llama-7b-hf", add_eos_token=True
    )
except Exception as e:
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        "decapoda-research/llama-7b-hf", add_eos_token=True
    )
model = prepare_model_for_int8_training(model)


config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
data = load_dataset("json", data_files="/data/lora/cn_alpaca.json")

# %%

train_val = data["train"].train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
train_data = train_val["train"]
val_data = train_val["test"]

# %%

train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x)))
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x)))

# %%
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=80,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=5,
        output_dir="/data/lora-cn",
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=20,
        save_steps=20,
        ddp_find_unused_parameters=False if ddp else None,
        load_best_model_at_end=True,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

# %%

trainer.train()

# %%

model.save_pretrained("/data/lora/lora-cal")


# %%
