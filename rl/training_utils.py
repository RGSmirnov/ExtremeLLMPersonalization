from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
import logging
from functools import partial
from transformers import BitsAndBytesConfig

'''
LoRA parameters are hardcoded here
'''

'''
For BitLoRA
'''
'''
import importlib
from bitlinear import BitLoraLayer

modeling_btlm = importlib.import_module("peft")
modeling_btlm.tuners.lora.layer.LoraLayer.update_layer = (  # pylint: disable=protected-access
    BitLoraLayer.update_layer
)
print("patched?")
'''

def get_quant_config(conf):
    if conf.peft:
        if any(["int8" in conf.base_quantization, "int4" in conf.base_quantization]):
            quantisation_config = BitsAndBytesConfig(
                load_in_8bit="int8" in conf.base_quantization,
                load_in_4bit="int4" in conf.base_quantization,
            )
            print(quantisation_config)
            return quantisation_config
    return None


def lora_prepare(model, gradient_checkpointing = True, use_reentrant_gradient_checkpointing = True, dora = True):
    peft_config = LoraConfig(target_modules='all-linear', use_dora=dora, r=32, lora_alpha=16, lora_dropout=0.05)
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant":use_reentrant_gradient_checkpointing})
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())
    return model

def load_model_tokenizer(config):
    model = AutoModelForCausalLM.from_pretrained(config.model_path, 
                                                 use_cache = False,
                                                 cache_dir = config.cache_dir,
                                                 #attn_implementation="flash_attention_2",
                                                 #torch_dtype = torch.float16,
                                                 quantization_config = get_quant_config(config),
                                                 trust_remote_code = True,
                                                 token = config.hf_token
                                                )
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code = True, token = config.hf_token)
    if config.gradient_checkpointing is True and not config.peft:
        # reentrant configs is set only in PEFT?
        model.gradient_checkpointing_enable()
    if len(config.special_tokens_to_add)>0:
        # add special tokens
        tokenizer.add_tokens(list(config.special_tokens_to_add))
        model.resize_token_embeddings(len(tokenizer))
    
    if config.peft is True:
        model.enable_input_require_grads() # can turn on checkpointing for the model alternatively before LoRA
        model = lora_prepare(model, config.gradient_checkpointing, config.use_reentrant_gradient_checkpointing, False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.unk_token is None else tokenizer.unk_token 
    #tokenizer.pad_token = tokenizer.unk_token # if it is EOS token we ruin labels - the problem is UNK and EOS sometime are the same...
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    tokenizer.max_length = config.max_length
    return model, tokenizer


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1,
    # added
    # it cannot be configured externaly
    max_lr_decrease_percentage: float = 0.9
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        max_lr_decrease_percentage = max_lr_decrease_percentage
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float,
    #modificated
    max_lr_decrease_percentage: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    #return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))*max_lr_decrease_percentage+(1-max_lr_decrease_percentage)
