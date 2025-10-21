# utils_qwen.py

from typing import List, Optional
import numpy as np
import torch
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND

class AttentionStore:
    @staticmethod
    def get_empty_store():
        # A transformer has a single sequence of blocks, so we simplify the store
        return {"transformer_block": []}

    def __call__(self, attn, is_cross: bool, place_in_transformer: str):
        # We are only interested in cross-attention maps
        if self.cur_att_layer >= 0 and is_cross:
            if attn.shape[1] == np.prod(self.attn_res):
                self.step_store[place_in_transformer].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        return self.attention_store

    def aggregate_attention(self) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads."""
        out = []
        attention_maps = self.get_average_attention()
        # We only have one place now: 'transformer_block'
        for item in attention_maps["transformer_block"]:
            cross_maps = item.reshape(-1, self.attn_res[0], self.attn_res[1], item.shape[-1])
            out.append(cross_maps)
        
        if not out:
             # Return a zero tensor if no attention maps were captured
            return torch.zeros(self.attn_res[0], self.attn_res[1], 77).to('cuda') # Assuming max_length 77

        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res):
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res

# The original AttnProcessor is often compatible if the Qwen attention module
# follows the standard diffusers API. We can try using it directly.
class AttnProcessor:
    def __init__(self, attnstore, place_in_transformer):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_transformer = place_in_transformer

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs
    ) -> torch.Tensor:
        # This is the standard forward call for a diffusers Attention processor.
        # It's kept from the original implementation as it's likely compatible.
        
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if not is_cross:
            encoder_hidden_states = hidden_states
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # Store the attention probabilities.
        self.attnstore(attention_probs, is_cross, self.place_in_transformer)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_attention_control(transformer, attention_store):
    """
    Adapted to register attention control for a Transformer model.
    """
    attn_procs = {}
    cross_att_count = 0
    
    # Iterate through the transformer blocks to find attention processors
    for name in transformer.attn_processors.keys():
        # We assume all cross-attention happens within the main transformer blocks
        # and we don't need to distinguish between up/mid/down.
        is_cross_attention = "attn2" in name or "cross_attn" in name
        
        if is_cross_attention:
            cross_att_count += 1
            attn_procs[name] = AttnProcessor(
                attnstore=attention_store,
                place_in_transformer="transformer_block"
            )

    if not attn_procs:
        print("Warning: No cross-attention processors found to attach CLoRA hooks.")
        
    transformer.set_attn_processor(attn_procs)
    attention_store.num_att_layers = cross_att_count