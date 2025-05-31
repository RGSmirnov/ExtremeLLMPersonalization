from torch import nn, Tensor
import torch.nn.functional as F
import math
from peft.tuners.lora.layer import LoraLayer
import torch
import torch.nn.functional as F
from torch import nn


def weight_quant(w: Tensor):
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u

def weight_quant_1d58(w: Tensor):
    absmean_weight = torch.mean(torch.abs(w))
    weight_adjustment_factor = 1e-4 + absmean_weight / 2
    input_tensor = w / weight_adjustment_factor
    return torch.clip(input=torch.round(input_tensor), min=-1, max=1)*absmean_weight
    

class BitLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        w = self.weight
        # Normalization here?

        # STE using detach
        x_quant = x
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y
    
    def forward_158(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        
        # converges much better without RMSNorm
        
        w = self.weight
        #absmean_weight = torch.mean(torch.abs(w))
        #weight_adjustment_factor = 1e-4 + absmean_weight / 2
        
        x_quant = x #*weight_adjustment_factor
        # here because of detaching we are training W and not weight_quant(w) - so the optimisation is full precision
        w_quant = w + (weight_quant_1d58(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y
    
    def forward_qa(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        # converges much better without RMSNorm
        # we want to merge it later without any issues
        x = self.quant(x)
        x = F.linear(x, self.weight)
        x = self.dequant(x)
        return x

class BitLoraLayer(LoraLayer):
    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = BitLinear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = BitLinear(r, self.out_features, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)
    
