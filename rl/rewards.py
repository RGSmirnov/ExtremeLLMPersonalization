import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import spacy
import regex

'''
The reference styles are hardcoded here
The dropout implementation is far not efficient - legacy from the early experiments with the nn.Dropout that also rescales the values 
'''

def restore_all_zeros(original, after_dropout):
    # no rescaling version
    # also avoiding all the styles similarity for the sample to be 0
    _, s = original.shape
    mask = torch.zeros((1, s))
    mask = torch.where(torch.sum(after_dropout, dim = -2)>0, 0, 1)
    dropout_mask = torch.where(after_dropout>0, 1, 0)
    return mask*original+dropout_mask*original

class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # Initialize the base transformer model.
        self.model = AutoModel.from_config(config).eval()
        # Define a classifier head.
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights (handled by PreTrainedModel)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the transformer
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Classifier
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output

class Rewards:
    def __init__(self, accelerator = None, log_metrics = None):
        self.accelerator = accelerator # for device and logging
        self.log_metrics = log_metrics
        
        ### AI Detection ###
        reward_model_directory = "desklib/ai-text-detector-v1.01"
        self.ai_detection_model = DesklibAIDetectionModel.from_pretrained(reward_model_directory).eval()
        self.ai_detection_tokenizer = AutoTokenizer.from_pretrained(reward_model_directory)
        self.ai_detection_model.to(torch.device('cuda'))
        self.max_len=768

        ### NER ###
        self.ner_pipeline = spacy.load("xx_ent_wiki_sm")

        ### Length ###
        self.length_allowed_delta = 30

        ### Format ###
        self.required_prefix = "Rewritten:"

        ### Style ###
        self.style_encoder = SentenceTransformer('StyleDistance/styledistance').to(torch.device('cuda'))
        self.style_examples = [
            "Good evening Sir or Madam, I would like to introduce myself.",
            "Lori's gonna rock being boss at HiTOPS; she'll seriously push things forward.", # informal# "Hey dude, what's up? Such a funny story I have in my mind to share",
            "Wow :-), I'll advocate for Blanco's dedication to ethical business, and CRT membership =D!" # Text emojies #"Telling the truth - I really hate it, if you do it one more time you will experience consequences!"
        ]
        with torch.no_grad():
            self.style_examples_encoded = self.style_encoder.encode(self.style_examples)
        
    def __name__(self):
        return "reward"

    def __call__(self, prompts, completions, **kwargs):
        dict_return = kwargs.get("return_dict", False)
        #print(completions)
        #print(prompts)
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(completions, str):
            completions = [completions]

        # they are list of lists of dicts
        if isinstance(prompts, list) and isinstance(prompts[0][0], dict):
            #print("HERE")
            prompts = [x[-1]['content'] for x in prompts]
        if isinstance(completions, list) and isinstance(completions[0][0], dict):
            #print("HERE")
            completions = [x[-1]['content'] for x in completions]
        
        # ideally can happen in parallel
        assert len(completions)==len(prompts)

        mask = torch.tensor([1 if self.required_prefix in x else 0 for x in completions]).to(self.ai_detection_model.device, self.ai_detection_model.dtype)
        completions = [x.split(self.required_prefix,1)[-1].strip() for x in completions]
        
        lenth_components = []
        for i,x in enumerate(completions):
            completion_length = len(x.split())
            prompt_length = len(prompts[i].split())
            if completion_length>prompt_length+self.length_allowed_delta:
                lenth_components.append(0.0)
            else:
                lenth_components.append(1.0)
        lenth_components = torch.tensor(lenth_components).squeeze().to(torch.device('cuda'))#(self.accelerator.device)

        # ai detection
        encoded = self.ai_detection_tokenizer(
            completions,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(torch.device('cuda'))#.to(device)
        attention_mask = encoded['attention_mask'].to(torch.device('cuda'))#.to(device)
        with torch.no_grad():
            outputs = self.ai_detection_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            probability = torch.sigmoid(logits).squeeze()
        reward = (1-probability)
        reward = torch.where(reward<0.5, reward, (reward**(1/2)*1.11).clip(0,1))

        # ner
        ner_rewards = []
        for inp, out in zip(prompts, completions):
            ner_rewards.append(self.ner_reward(inp,out))
        ner_rewards = torch.tensor(ner_rewards).squeeze().to(torch.device('cuda'))#.to(self.accelerator.device)

        # style
        with torch.no_grad():
            completion_styles = self.style_encoder.encode(completions)
        cos = cos_sim(self.style_examples_encoded, completion_styles)
        #print(cos)
        # For reward we can firstly do dropout to force exploration
        dropout = torch.nn.Dropout(0.2)
        with torch.no_grad():   
            dropout_cos = dropout(cos)
        dropout_cos = restore_all_zeros(cos, dropout_cos)
        # because of the encoder specifics majority of similarities are higher than 0.7, so the similarity changes from 0.7 to 1
        # do rescaling because of this
        style = (torch.max(dropout_cos, dim=0).values.to(torch.device('cuda')).clip(0.7,1)-0.7)*3.5
        # balancer
        balancer = torch.std(cos, dim=0).to(torch.device('cuda'))
        sft_inner = torch.nn.Softmax(dim = 0)
        sft_outer = torch.nn.Softmax(dim = 1)
        e = 1e-10
        p_inner = sft_inner(cos+e)
        p_outer = sft_outer(cos+e)
        std_inner = torch.std(p_inner, dim = 0)
        std_outer = torch.std(p_outer, dim = 1)
        if self.accelerator:
            self.log_metrics({
                "inner_style_sample_std": torch.mean(std_inner.to(torch.float32).to(torch.device("cpu"))).item(),
                "outer_samples_style_std": torch.mean(std_outer.to(torch.float32).to(torch.device("cpu"))).item(),
                "ner_reward":torch.mean(ner_rewards.to(torch.float32).to(torch.device("cpu"))).item(), 
                "mask": torch.mean(mask.to(torch.float32).to(torch.device("cpu"))).item(), 
                "detector_reward": torch.mean(reward.to(torch.float32).to(torch.device("cpu"))).item(), 
                # probably later we should do style reward monitoring without dropout here
                "style_reward": torch.mean(style.to(torch.float32).to(torch.device("cpu"))).item(), 
                "balancer": torch.mean(balancer.to(torch.float32).to(torch.device("cpu"))).item(), 
                "lenth_component": torch.mean(lenth_components.to(torch.float32).to(torch.device("cpu"))).item()})
        else:
            print({
                "inner_style_sample_std": torch.mean(std_inner.to(torch.float32).to(torch.device("cpu"))).item(),
                "outer_samples_style_std": torch.mean(std_outer.to(torch.float32).to(torch.device("cpu"))).item(),
                "ner_reward":torch.mean(ner_rewards.to(torch.float32).to(torch.device("cpu"))).item(), 
                "mask": torch.mean(mask.to(torch.float32).to(torch.device("cpu"))).item(), 
                "detector_reward": torch.mean(reward.to(torch.float32).to(torch.device("cpu"))).item(), 
                "style_reward": torch.mean(style.to(torch.float32).to(torch.device("cpu"))).item(), 
                "balancer": torch.mean(balancer.to(torch.float32).to(torch.device("cpu"))).item(), 
                "lenth_component": torch.mean(lenth_components.to(torch.float32).to(torch.device("cpu"))).item()})
        if dict_return:
            return {
                "inner_style_sample_std": torch.mean(std_inner.to(torch.float32).to(torch.device("cpu"))).item(),
                "outer_samples_style_std": torch.mean(std_outer.to(torch.float32).to(torch.device("cpu"))).item(),
                "ner_reward":torch.mean(ner_rewards.to(torch.float32).to(torch.device("cpu"))).item(), 
                "mask": torch.mean(mask.to(torch.float32).to(torch.device("cpu"))).item(), 
                "detector_reward": torch.mean(reward.to(torch.float32).to(torch.device("cpu"))).item(), 
                "style_reward": torch.mean(style.to(torch.float32).to(torch.device("cpu"))).item(), 
                "balancer": torch.mean(balancer.to(torch.float32).to(torch.device("cpu"))).item(), 
                "lenth_component": torch.mean(lenth_components.to(torch.float32).to(torch.device("cpu"))).item(),
                "cosines": cos.squeeze().tolist()
            }
        # balancer should be implement somehow else
        return mask*(reward+style+lenth_components+ner_rewards) #style #ner_rewards*mask*(reward+style+lenth_components)
    
    def ner_reward(self, x_text, y_text):
        # make it word-wise, no punct
        doc = " ".join([regex.sub(r'\p{P}+', ' ', x.text) for x in self.ner_pipeline(x_text).ents])
        doc_1 = " ".join([regex.sub(r'\p{P}+', ' ', x.text) for x in self.ner_pipeline(y_text).ents])
        missed = 0
        num_regex = 0
        for ent in doc.split():
            num_regex+=1
            if ent not in doc_1:
                missed += 1
        return 1-missed/num_regex if num_regex else 1
