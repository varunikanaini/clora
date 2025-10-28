from typing import List
import torch

class QwenAttentionStore:
    def __init__(self, attn_res=(64, 64)):
        self.attn_res = attn_res
        self.reset()

    def reset(self):
        self.step_store = []
        self.attention_store = []

    def __call__(self, attn):
        self.step_store.append(attn.squeeze(0))

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = []

    def aggregate_attention(self) -> torch.Tensor:
        if not self.attention_store:
            return torch.zeros((self.attn_res[0] * self.attn_res[1]), 128, device='cuda')
        
        maps = torch.stack(self.attention_store, dim=0).mean(dim=[0, 1])
        return maps # Return shape [h*w, tokens]

def register_attention_control(pipeline, attention_store: QwenAttentionStore):
    TARGET_ATTENTION_CLASS = "Attention"
    def get_attn_hook(store):
        def attn_hook(module, a_input, a_output):
            if len(a_input) > 1 and a_input[1] is not None:
                store(a_output[1].detach()) # Detach for now, we don't need grads for this step
        return attn_hook
    print(f"Registering hooks on modules of class: '{TARGET_ATTENTION_CLASS}'")
    for name, module in pipeline.transformer.named_modules():
        if module.__class__.__name__ == TARGET_ATTENTION_CLASS:
            module.register_forward_hook(get_attn_hook(attention_store))