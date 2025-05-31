# ExtremeLLMPersonalization

This repository contains code to implement the main parts of the attached paper "Extreme multi-style LLM personalization with rewardsdropout"

Abstract from the paper: 

`In this paper, we address the problem of multiple correct behaviors, which is critical for achieving extreme individual user personalization. We propose a novel solution based on the reward dropout method. Using the task of style transfer designed to bypass AI detection — where multiple valid output styles are allowed — we demonstrate that reward dropout not only effectively handles the multiple correct behavior challenge but also serves as a regularization technique for the reinforcement learning (RL) training process, analogous to the original dropout used in neural networks. Our approach emphasizes personalization through the use of LoRA adapters, enabling a more flexible and resource-efficient personalization framework. Furthermore, we investigate the Bit-LoRA method and show that, despite its limited exploration capacity, it can still enhance the efficiency of the personalization system.`

---

Open source plan:
- [x] release main code and data
- [ ] opensource model weights
- [ ] improve the components

---

Folders:

Warmup - contains data for the warmup stage of training (SFT)

RL - contains main components for Dr.GRPO training with rewards dropout and Bit-LoRA components

Inferece_eval - contains Jupyter notebook with the main inference and evaluation components used during the published research

