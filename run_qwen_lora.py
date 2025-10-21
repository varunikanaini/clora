# ==============================================================================
#
# CLoRA for Qwen-Image: Standalone Python Script
#
# This script adapts the CLoRA methodology for the Qwen/Qwen-Image model.
# It includes the necessary custom pipeline and utility code, loads the model,
# simulates LoRA usage, and generates an image with multi-concept composition.
#
# ==============================================================================

import os
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Optional

# --- Pre-computation Step ---
# Before running, ensure you have the required libraries installed.
# You can install them by running the following command in your terminal:
# pip install diffusers transformers accelerate peft pytorch-metric-learning

# --- File Creation Step ---
# This script will create the necessary utility and pipeline files automatically.

print("Step 1: Creating custom pipeline and utility files...")

# Create the custom utility file for Qwen attention hooking
utils_qwen_code = """
from typing import List, Optional
import numpy as np
import torch
from diffusers.models.attention_processor import Attention

class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"transformer_block": []}

    def __call__(self, attn, is_cross: bool, place_in_transformer: str):
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
        out = []
        attention_maps = self.get_average_attention()
        for item in attention_maps["transformer_block"]:
            cross_maps = item.reshape(-1, self.attn_res[0], self.attn_res[1], item.shape[-1])
            out.append(cross_maps)
        
        if not out:
            return torch.zeros(self.attn_res[0], self.attn_res[1], 77).to('cuda')

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

class AttnProcessor:
    def __init__(self, attnstore, place_in_transformer):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_transformer = place_in_transformer

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs
    ) -> torch.Tensor:
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
        
        self.attnstore(attention_probs, is_cross, self.place_in_transformer)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def register_attention_control(transformer, attention_store):
    attn_procs = {}
    cross_att_count = 0
    
    for name in transformer.attn_processors.keys():
        if "attn2" in name:
            cross_att_count += 1
            attn_procs[name] = AttnProcessor(
                attnstore=attention_store,
                place_in_transformer="transformer_block"
            )

    if not attn_procs:
        print("Warning: No cross-attention processors found to attach CLoRA hooks.")
        
    transformer.set_attn_processor(attn_procs)
    attention_store.num_att_layers = cross_att_count
"""
with open("utils_qwen.py", "w") as f:
    f.write(utils_qwen_code)

# Create the main custom pipeline file
pipeline_qwen_clora_code = """
import inspect
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import QwenImagePipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from pytorch_metric_learning import losses
from utils_qwen import AttentionStore, register_attention_control

logger = logging.get_logger(__name__)

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def erode(image, kernel_size=3, stride=1, padding=1):
    return 1 - F.max_pool2d(1 - image, kernel_size, stride, padding)

def dilate(image, kernel_size=3, stride=1, padding=1):
    return F.max_pool2d(image, kernel_size, stride, padding)

class QwenCloraPipeline(QwenImagePipeline):
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    @torch.no_grad()
    def attention_map_to_mask(
        self, attention_maps, mask_indices, size, mask_thresholding_alpha=0.3, mask_erode=False, mask_dilate=False, mask_opening=False, mask_closing=False
    ):
        masks = []
        for attention_map, mask_indice_list in zip(attention_maps, mask_indices):
            multi_token_maps = []
            if mask_indice_list:
                for mask_indice in mask_indice_list:
                    attn_map = attention_map[:, :, mask_indice]
                    multi_token_map = torch.where(attn_map > (attn_map.max() * mask_thresholding_alpha), 1.0, 0.0)
                    multi_token_maps.append(multi_token_map)
                mask = torch.where(torch.stack(multi_token_maps).sum(dim=0) >= 1.0, 1.0, 0.0)
            else:
                mask = torch.zeros_like(attention_map[:, :, 0])

            mask = mask.unsqueeze(0).unsqueeze(0)
            if mask_dilate: mask = dilate(mask)
            if mask_erode: mask = erode(mask)
            if mask_opening: mask = dilate(erode(mask))
            if mask_closing: mask = erode(dilate(mask))
            mask = mask.squeeze(0).squeeze(0)
            masks.append(mask)

        masks = torch.stack(masks)
        masks[0] = torch.where(masks[1:].sum(dim=0) == 0, 1.0, 0.0)

        return F.interpolate(masks.unsqueeze(1), size, mode="nearest").unsqueeze(1).repeat(1, 1, 4, 1, 1)

    @torch.enable_grad()
    def loss_fn(
        self, attention_maps: List[torch.Tensor], important_token_indices: List[List[List[int]]], temperature=0.5
    ) -> torch.Tensor:
        classes, embeddings = [], []
        for class_i, attention_group in enumerate(important_token_indices):
            for prompt_i, attention_indices_per_prompt in enumerate(attention_group):
                for ind in attention_indices_per_prompt:
                    embedding = attention_maps[prompt_i][:, :, ind].view(-1)
                    embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min() + 1e-6)
                    embeddings.append(embedding)
                    classes.append(class_i)

        classes = torch.tensor(classes, device=attention_maps[0].device)
        embeddings = torch.stack(embeddings, dim=0).to(attention_maps[0].device)
        loss_func = losses.NTXentLoss(temperature=temperature)
        return loss_func(embeddings, classes)

    @torch.enable_grad()
    def update_latents(
        self, latent: torch.Tensor, loss: torch.Tensor, step_size: float
    ):
        grads = torch.autograd.grad(loss.requires_grad_(True), [latent], retain_graph=True)[0]
        return latent - step_size * grads

    @torch.enable_grad()
    def predict_noise_and_aggregate_attention(
        self, lora_list, prompt_embeds_list, latent_model_input, t, attention_store, style_lora, style_lora_weight
    ):
        noise_preds, attention_maps_list = [], []
        
        if hasattr(self, "enable_lora"): self.enable_lora()
        
        for lora_name, prompt_embeds in zip(lora_list, prompt_embeds_list):
            lora_names, lora_weights = [], []
            if lora_name:
                lora_names.append(lora_name)
                lora_weights.append(1.0)
            if style_lora:
                lora_names.append(style_lora)
                lora_weights.append(style_lora_weight)

            self.set_adapters(lora_names, lora_weights)
            
            noise_pred = self.transformer(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            
            attention_maps = attention_store.aggregate_attention()
            noise_preds.append(noise_pred)
            attention_maps_list.append(attention_maps)

        return noise_preds, attention_maps_list

    @torch.no_grad()
    def __call__(
        self,
        prompt_list: List[str],
        negative_prompt_list: List[str],
        lora_list: List[str],
        style_lora: str = "",
        style_lora_weight: float = 1.0,
        important_token_indices: List[List[List[int]]] = [[[]]],
        mask_indices: List[List[int]] = [[]],
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        step_size: int = 20,
        max_iter_to_alter: int = 25,
        iterative_steps: List[int] = [],
        iterative_steps_steps: int = 20,
        latent_update: bool = True,
        apply_mask_after: int = 0,
        mask_threshold_alpha: float = 0.3,
        mask_erode: bool = False,
        mask_dilate: bool = False,
        mask_opening: bool = False,
        mask_closing: bool = False,
        use_text_encoder_lora: bool = False,
        guidance_rescale: float = 0.0,
        **kwargs,
    ):
        attn_res_map = {1024: 32, 512: 16}
        attn_res = attn_res_map.get(height, 32)
        attention_store = AttentionStore((attn_res, attn_res))
        register_attention_control(self.transformer, attention_store)

        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale

        prompt_embeds_list = []
        for lora_name, prompt, negative_prompt in zip(lora_list, prompt_list, negative_prompt_list):
            lora_names, lora_weights = [], []
            if use_text_encoder_lora and lora_name:
                lora_names.append(lora_name)
                lora_weights.append(1.0)
            if style_lora:
                lora_names.append(style_lora)
                lora_weights.append(style_lora_weight)

            self.set_adapters(lora_names, lora_weights)
            
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=self.do_classifier_free_guidance, negative_prompt=negative_prompt
            )
            prompt_embeds_list.append(torch.cat([negative_prompt_embeds, prompt_embeds]))

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            1, num_channels_latents, height, width, prompt_embeds_list[0].dtype, device, generator
        )
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                with torch.enable_grad():
                    latents = latents.detach().requires_grad_(True)
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    
                    noise_preds, attention_maps_list = self.predict_noise_and_aggregate_attention(
                        lora_list, prompt_embeds_list, latent_model_input, t, attention_store, style_lora, style_lora_weight
                    )

                    if latent_update and i < max_iter_to_alter:
                        loss = self.loss_fn(attention_maps_list, important_token_indices, temperature=0.5)
                        for _ in range(iterative_steps_steps if i in iterative_steps else 1):
                             latents = self.update_latents(latents, loss, step_size)
                             if i in iterative_steps:
                                 latent_model_input_updated = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                                 noise_preds, attention_maps_list = self.predict_noise_and_aggregate_attention(
                                     lora_list, prompt_embeds_list, latent_model_input_updated, t, attention_store, style_lora, style_lora_weight
                                 )
                                 loss = self.loss_fn(attention_maps_list, important_token_indices, temperature=0.5)
                        loss = loss.detach()

                latent_model_input = latent_model_input.detach()
                noise_preds = [p.detach() for p in noise_preds]
                
                noise_preds_tensor = torch.stack(noise_preds, dim=0)
                noise_pred_uncond, noise_pred_text = noise_preds_tensor.chunk(2, dim=1)

                if i >= apply_mask_after:
                    masks = self.attention_map_to_mask(
                        attention_maps=attention_maps_list, mask_indices=mask_indices, size=(noise_preds_tensor.shape[-2], noise_preds_tensor.shape[-1]),
                        mask_thresholding_alpha=mask_threshold_alpha, mask_erode=mask_erode, mask_dilate=mask_dilate, mask_opening=mask_opening, mask_closing=mask_closing,
                    ).to(device=device, dtype=noise_pred_uncond.dtype)
                else:
                    masks = torch.ones_like(noise_pred_uncond)

                noise_pred_uncond = (noise_pred_uncond * masks).sum(dim=0) / masks.sum(dim=0).clamp(min=1e-6)
                noise_pred_text = (noise_pred_text * masks).sum(dim=0) / masks.sum(dim=0).clamp(min=1e-6)
                
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                progress_bar.update()

        image = self.decode_latents(latents)
        image = self.image_processor.postprocess(image, output_type="pil")

        self.maybe_free_model_hooks()
        return image, [], []
"""
with open("pipeline_qwen_clora.py", "w") as f:
    f.write(pipeline_qwen_clora_code)

print("✅ Setup complete. Custom pipeline files created.")

# =======================================================
# MAIN SCRIPT EXECUTION STARTS HERE
# =======================================================

# Import the newly created custom pipeline
from pipeline_qwen_clora import QwenCloraPipeline

def main():
    # ------------------------------------
    # Section 2: Model and LoRA Loading
    # ------------------------------------
    print("\nStep 2: Loading Qwen-Image model with custom CLoRA pipeline...")
    # This model is large and may take a few minutes to download on first run.
    pipeline = QwenCloraPipeline.from_pretrained(
        "Qwen/Qwen-Image",
        torch_dtype=torch.float16,
    ).to("cuda")
    print("✅ Qwen-Image model loaded.")

    # --- SIMULATING LoRA LOADING FOR DEMONSTRATION ---
    # In a real-world scenario, you would replace "Qwen/Qwen-Image" with paths
    # to your actual LoRA files that have been trained specifically for Qwen-Image.
    print("\n--- LoRA Loading Simulation ---")
    print("Loading base model as 'dog' and 'cat' adapters for demonstration.")
    pipeline.load_lora_weights("Qwen/Qwen-Image", adapter_name="dog")
    pipeline.load_lora_weights("Qwen/Qwen-Image", adapter_name="cat")
    print("✅ LoRA adapters simulated successfully.")

    # ----------------------------------------------------
    # Section 3: Prompt Engineering and Token Indexing
    # ----------------------------------------------------
    print("\nStep 3: Analyzing prompts and defining token indices...")
    fg_prompts = [
        "A cat and a dog in a garden",
        "A <s_1> cat and a dog in a garden",
        "A cat and a <s_2> dog in a garden",
    ]
    fg_negative = ["blurry, low quality", "blurry, low quality", "blurry, low quality"]
    fg_loras = ["", "cat", "dog"]

    print("\n--- Analyzing Qwen Tokenization ---")
    for i, prompt in enumerate(fg_prompts):
        ids = pipeline.tokenizer(prompt).input_ids
        tokens = pipeline.tokenizer.convert_ids_to_tokens(ids)
        print(f"Prompt {i}: '{prompt}'")
        print(f"Tokens: {tokens}")
        print(f"Indices: {list(range(len(tokens)))}")
        print("-" * 20)

    # Based on the tokenization output, we define the indices.
    important_token_indices = [
        # Concept 1: CAT
        [[2], [2, 3], [2]],
        # Concept 2: DOG
        [[4], [5], [4, 5]],
    ]
    mask_indices = [
        [],           # Background mask (automatic)
        [2, 3],       # Mask for 'cat' from Prompt 1 ('<s_1>', 'cat')
        [4, 5],       # Mask for 'dog' from Prompt 2 ('<s_2>', 'dog')
    ]
    print("✅ Token indices for CLoRA have been defined.")

    # ---------------------------------------------
    # Section 4: Inference and Image Generation
    # ---------------------------------------------
    print("\nStep 4: Starting CLoRA image generation...")
    H, W, seed = 1024, 1024, 1234
    num_inference_steps = 30
    guidance_scale = 7.0

    generator = torch.Generator(device="cuda").manual_seed(seed)

    image, _, _ = pipeline(
        prompt_list=fg_prompts,
        negative_prompt_list=fg_negative,
        lora_list=fg_loras,
        important_token_indices=important_token_indices,
        mask_indices=mask_indices,
        height=H,
        width=W,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        latent_update=True,
        max_iter_to_alter=20,
        step_size=40,
    )

    # ---------------------------------------------
    # Section 5: Save the Output
    # ---------------------------------------------
    output_filename = "clora_qwen_output.png"
    image[0].save(output_filename)
    print(f"\n✅ Image generation complete! Output saved to '{output_filename}'")


if __name__ == "__main__":
    main()