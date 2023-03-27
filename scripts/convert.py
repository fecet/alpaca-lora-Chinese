# %%
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

import torch

device_map = "auto"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000

# %%
from pathlib import Path

def get_basemodel():
    model = AutoModelForCausalLM.from_pretrained(
        "decapoda-research/llama-13b-hf",
        cache_dir="data/hf",
        load_in_8bit=True,
        device_map=device_map,
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
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    return model

def convert_to_lora(ckpt_path, model):
    params_dir = Path(ckpt_path) / "pytorch_model.bin"
    model.load_state_dict(torch.load(params_dir, map_location="cpu"))
    # trainer.train()
    model.save_pretrained(Path("/data/lora/") / ckpt_path.name)

# %%
model=get_basemodel()
# %%

convert_to_lora(Path("data/checkpoint-8100"),model)
