import torch.nn as nn
import numpy as np

class LoRALinear(nn.Module):
    def __init__(self, orig_linear, rank=4, alpha=1.0, device=None, dtype=None):
        super().__init__()
        self.orig = orig_linear
        
        # Inherit device/dtype from original linear layer if not provided
        device = device or orig_linear.weight.device
        dtype = dtype or orig_linear.weight.dtype

        self.lora_down = nn.Linear(orig_linear.in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_up = nn.Linear(rank, orig_linear.out_features, bias=False, device=device, dtype=dtype)
        self.scaling = alpha / rank
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_down.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.orig(x) + self.lora_up(self.lora_down(x)) * self.scaling
