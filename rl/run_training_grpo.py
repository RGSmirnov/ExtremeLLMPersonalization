import fire
from dataclasses import dataclass
import json

from data_utils import get_raw_dataset
from training_utils import load_model_tokenizer
import logging
import random
import torch

from transformers.trainer_pt_utils import get_parameter_names
import bitsandbytes as bnb
from torch import nn

from trl import GRPOConfig, GRPOTrainer

from rewards import Rewards

logging.basicConfig(level=logging.DEBUG)


class CustomLoggerGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_funcs[0].accelerator = self.accelerator
        self.reward_funcs[0].log_metrics = self.log


@dataclass
class DefaultConfigs():
    model_path: str
    train_datasets: list
    test_datasets: list = None
    shuffle: bool = True
    peft: bool = True
    base_quantization: str = "None"#'int8'
    mixed_precision: str = 'bf16'
    fp16: bool = False
    bf16: bool = True
    flash_attention: bool = True
    scheduler_max_decrease: float = 0.9
    max_length: int = 1024
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 32 # 1 if not Peft
    per_device_eval_batch_size: int = 32 # 8 if not peft # should meet the num generations condition (accum x per_device_train_batch_size)
    warmup_steps_ratio: float = 0.0
    cache_dir: str = "cache_new/huggingface/hub/" # None
    warmup_strategy: str = 'relative'
    output_dir: str = "model_peft_1"
    gradient_accumulation_steps: int = 1#4#1 # 8 if not peft
    learning_rate: float = 1e-5 #1e-6#1e-4#1e-5#1e-4#1e-6 # 1e-6 if not peft
    save_safetensors: bool = False
    gradient_checkpointing: bool = True
    use_reentrant_gradient_checkpointing: bool = False
    report_to: str = 'wandb'
    label_smoothing_factor: float = 0.0 # does our trainer support it?
    train_rate: float = 0.9 # if no eval dataset
    full_determinism: bool = False 
    evaluation_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_strategy: str = "steps"
    save_steps: int = 100
    logging_steps:int = 1
    prediction_loss_only:bool = True
    max_grad_norm: float = 1.0
    train_sample_packing: bool = False
    eval_sample_packing: bool = False
    special_tokens_to_add: tuple = ()
    text_formatting_style: str = 'llama3'
    add_system_default: bool = False
    do_eval_first: bool = False
    optimizer_name: str = "8bitadam"
    hf_token: str = ",,,"
    # GRPO-related
    logging_steps: int = 1
    num_generations: int = 8 #ValueError: The effective train batch size (1 x 1 x 4) must be evenly divisible by the number of generations per prompt (8). Given the current effective train batch size, the valid values for the number of generations are: [2, 4].
    max_prompt_length: int = 312
    max_completion_length: int = 312
    loss_type: str = "dr_grpo"
    mask_truncated_completions: bool = True
    beta: float = 0.0  # NO KL IF 0.0 # WHEN 0.04 (default) - not converging for this task
    temperature: float = 1.5 # make generations more diverse

    def convert_dataclass(self, transfer_dataclass):        
        return transfer_dataclass(**{k:self.__dict__[k] for k,_ in transfer_dataclass.__dataclass_fields__.items() if k in self.__dict__})



def run(**config):
    # wandb
    # deepspeed
    # private folders in HF - llama
    with open(config['path'], 'r') as file:
        configs = json.loads(file.read())
    config = DefaultConfigs(**configs)
    print("DefaultConfigs")
    print(config)
    training_configs = config.convert_dataclass(GRPOConfig)
    print("training_configs")
    print(training_configs)
    # train / test
    dataset = get_raw_dataset(config.train_datasets)
    if config.test_datasets is not None:
        train_dataset = dataset
        test_dataset = get_raw_dataset(config.test_datasets)
    else:
        if config.full_determinism is False:
            random.shuffle(dataset)
        train_dataset = dataset[:int(config.train_rate*len(dataset))]
        test_dataset = dataset[int(config.train_rate*len(dataset)):]

    train_dataset = [{"prompt":x} for x in train_dataset] # better check if it adds generation prompt
    test_dataset = [{"prompt":x} for x in test_dataset] # better check if it adds generation prompt
    model, tokenizer = load_model_tokenizer(config)
    
    print("Doing training sampling")
    
    
    if config.warmup_strategy == 'relative':
        # this should be adjusted with gradient_accumulation_steps 
        training_configs.warmup_steps = int(len(train_dataset)*config.warmup_steps_ratio/(training_configs.per_device_train_batch_size*training_configs.world_size*config.gradient_accumulation_steps))
    else:
        training_configs.warmup_steps = config.warmup_steps_ratio
    model.is_parallelizable = True
    model.model_parallel = True
    training_configs.remove_unused_columns = False
    # we can also pass optimizer and lr_scheduler to the trainer, but it is not working with DeepSpeed / FSDP (???)
    
    # 8bit adam optimiser
    if config.optimizer_name == '8bitadam':
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                "weight_decay": training_configs.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optimizer_kwargs = {
            "betas": (training_configs.adam_beta1, training_configs.adam_beta2),
            "eps": training_configs.adam_epsilon,
        }
        optimizer_kwargs["lr"] = training_configs.learning_rate
        optim = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            betas=(training_configs.adam_beta1, training_configs.adam_beta2),
            eps=training_configs.adam_epsilon,
            lr=training_configs.learning_rate,
        )
    elif config.optimizer_name == 'adamw':
        optim = torch.optim.AdamW(
            model.parameters(),
            betas=(training_configs.adam_beta1, training_configs.adam_beta2),
            eps=training_configs.adam_epsilon,
            lr=training_configs.learning_rate,
        )
    training_configs.label_names = ['labels'] # important for eval 
    
    reward_function = Rewards()
    
    trainer = CustomLoggerGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_function,
        args=training_configs,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        optimizers=(optim, None)
    )
    
    if config.do_eval_first is True:
        trainer.evaluate()
    trainer.train()
    trainer.save_model(training_configs.output_dir+"_adapter")
    if config.peft is True:
        trainer.model = trainer.model.merge_and_unload()
    trainer.save_model(training_configs.output_dir)

if __name__=='__main__':
    fire.Fire(run)
