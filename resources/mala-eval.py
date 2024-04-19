import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from tqdm.auto import tqdm
import csv

BASE_PROMPT = """Below is an interaction between a human and an AI fluent in English and Amharic, providing reliable and informative answers. The AI is supposed to answer test questions from the human with short responses saying just the answer and nothing else.

Human: {instruction}

Assistant [Amharic] : """

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import datasets
import csv
from tqdm.auto import tqdm
from tqdm._tqdm_notebook import tqdm_notebook
import pandas as pd
tqdm_notebook.pandas()
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline


# Use a pipeline as a high-level helper
from transformers import pipeline



from typing import Optional, Any

import torch

from transformers.utils import is_accelerate_available, is_bitsandbytes_available
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline,
)

from peft import PeftModel

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def load_adapted_hf_generation_pipeline(
    base_model_name,
    lora_model_name,
    temperature: float = 0,
    top_p: float = 1.,
    max_tokens: int = 50,
    batch_size: int = 16,
    device: str = "cuda",
    load_in_8bit: bool = True,
    generation_kwargs: Optional[dict] = None,
):
    """
    Load a huggingface model & adapt with PEFT.
    Borrowed from https://github.com/tloen/alpaca-lora/blob/main/generate.py
    """

    if device == "cuda":
        if not is_accelerate_available():
            raise ValueError("Install `accelerate`")
    if load_in_8bit and not is_bitsandbytes_available():
            raise ValueError("Install `bitsandbytes`")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    task = "text-generation"
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.resize_token_embeddings(260164)
        model = PeftModel.from_pretrained(
            model,
            lora_model_name,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_model_name,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_in_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    generation_kwargs = generation_kwargs if generation_kwargs is not None else {}
    config = GenerationConfig(
        max_new_tokens=100,  # The maximum numbers of tokens to generate
        seed=42,  # seed value for reproducibility
        do_sample=True,  # Whether or not to use sampling; use greedy decoding otherwise.
        min_length=None,  # The minimum length of the sequence to be generated
        use_cache=True,  # [optional] Whether or not the model should use the past last key/values attentions
        top_p=1.0,  # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature=1.0,  # [optional] The value used to modulate the next token probabilities.
        top_k=5,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty=5.0,  # The parameter for repetition penalty. 1.0 means no penalty.
        length_penalty=1,  # [optional] Exponential penalty to the length used with beam-based generation.
        enable_azure_content_safety=False,  # Enable safety check with Azure content safety API
        enable_sensitive_topics=False,  # Enable check for sensitive topics using AuditNLG APIs
        enable_saleforce_content_safety=False,
            **generation_kwargs,
    )
    pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        batch_size=4, # TODO: make a parameter
        generation_config=config,
        framework="pt",
    )

    return pipe


pipe = load_adapted_hf_generation_pipeline(
    base_model_name="daryl149/llama-2-7b-hf",
    lora_model_name="MaLA-LM/mala-500",
    top_p=1.0,  # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature=1.0,  # [optional] The value used to modulate the next token probabilities.
    )
hf_dataset = datasets.load_dataset("israel/JOPUjJHxWmI5x",split='test[:4]') #['test']
BASE_PROMPT = """Below is an interaction between a human and an AI fluent in English and Amharic, providing reliable and informative answers. The AI is supposed to answer test questions from the human with short responses saying just the answer and nothing else.

Human: {instruction}

Assistant [Amharic] : """
def f(batch):
    return {'response':[pipe(BASE_PROMPT.format(instruction=f"{instruction}\n{input}"))[0]['generated_text'][len(BASE_PROMPT.format(instruction=f"{instruction}\n{input}")):] for instruction,input in zip(batch['instruction'],batch['input'])]}
    
hf_dataset = hf_dataset.map(lambda batch:f(batch) ,batch_size=8, batched=True)


hf_dataset.to_pandas().to_csv('mala-mala.csv',index=False)
