To do Bit-LoRA approach - need to uncomment in training_utils.py

---
Training:
```
pip3 install torch==2.6.0 torchvision
pip3 install transformers==4.51.3
pip3 install numpy==2.2.5 sentencepiece accelerate trl wandb fire peft deepspeed bitsandbytes
pip3 install flash-attn
pip3 install -U pip setuptools wheel
pip3 install -U spacy
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm
pip3 install regex
pip3 install sentence_transformers
wandb login ...
accelerate launch run_training_grpo.py --path training_configs_grpo.json
```
