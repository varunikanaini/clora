# pipeline_qwen_clora.py

import inspect
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import QwenImagePipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import logging, deprecate, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import randn_tensor
from pytorch_metric_learning import losses

# We will create a new utils_qwen.py file for our adapted attention mechanism
from utils_qwen import AttentionStore, register_attention_control

logger = logging.get_logger(__name__)

# Helper functions from the original pipeline_clora.py (can be kept as they are)
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def retrieve_timesteps(
    scheduler, num_inference_steps: Optional[int] = None, device: Optional[Union[str, torch.device]] = None, timesteps: Optional[List[int]] = None, **kwargs
):
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def erode(image, kernel_size=3, stride=1, padding=1):
    return 1 - F.max_pool2d(1 - image, kernel_size, stride, padding)

def dilate(image, kernel_size=3, stride=1, padding=1):
    return F.max_pool2d(image, kernel_size, stride, padding)


class QwenCloraPipeline(QwenImagePipeline):
    """
    CLoRA-adapted pipeline for the Qwen-Image model.
    """
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
        masks[0] = torch.where(masks.sum(dim=0) == 0, 1.0, masks[0])

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
                    embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min())
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
        
        # Ensure LoRA is enabled for the transformer
        if hasattr(self, "enable_lora"): self.enable_lora()
        
        for lora_name, prompt_embeds in zip(lora_list, prompt_embeds_list):
            lora_names, lora_weights = [], []
            if lora_name:
                lora_names.append(lora_name)
                lora_weights.append(1.0)
            if style_lora:
                lora_names.append(style_lora)
                lora_weights.append(style_lora_weight)

            # Set adapters for the current iteration
            self.set_adapters(lora_names, lora_weights)
            
            # Call the TRANSFORMER instead of the UNET
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
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        step_size: int = 20,
        max_iter_to_alter: int = 30,
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
        # 0. Set up attention control
        attn_res = (int(np.ceil(height / self.vae_scale_factor / 8)), int(np.ceil(width / self.vae_scale_factor / 8)))
        attention_store = AttentionStore(attn_res)
        register_attention_control(self.transformer, attention_store)

        # 1. Define call parameters
        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale

        # 2. Encode input prompts for each LoRA
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
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
            )
            prompt_embeds_list.append(torch.cat([negative_prompt_embeds, prompt_embeds]))

        # 3. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            1, num_channels_latents, height, width, prompt_embeds.dtype, device, generator
        )
        
        # 5. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                with torch.enable_grad():
                    latents = latents.detach().requires_grad_(True)
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    
                    self.transformer.zero_grad()
                    
                    noise_preds, attention_maps_list = self.predict_noise_and_aggregate_attention(
                        lora_list, prompt_embeds_list, latent_model_input, t, attention_store, style_lora, style_lora_weight
                    )

                    if latent_update and i < max_iter_to_alter:
                        loss = self.loss_fn(attention_maps_list, important_token_indices, temperature=0.5)
                        for _ in range(iterative_steps_steps if i in iterative_steps else 1):
                             latents = self.update_latents(latents, loss, step_size)
                             # Re-predict noise after update if iterating
                             if i in iterative_steps:
                                 latent_model_input_updated = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                                 latent_model_input_updated = self.scheduler.scale_model_input(latent_model_input_updated, t)
                                 noise_preds, attention_maps_list = self.predict_noise_and_aggregate_attention(
                                     lora_list, prompt_embeds_list, latent_model_input_updated, t, attention_store, style_lora, style_lora_weight
                                 )
                                 loss = self.loss_fn(attention_maps_list, important_token_indices, temperature=0.5)
                        loss = loss.detach()

                latent_model_input = latent_model_input.detach()
                noise_preds = [p.detach() for p in noise_preds]
                attention_maps_list = [a.detach() for a in attention_maps_list]

                noise_preds_tensor = torch.stack(noise_preds, dim=0)
                noise_pred_uncond, noise_pred_text = noise_preds_tensor.chunk(2, dim=1)

                if i >= apply_mask_after:
                    masks = self.attention_map_to_mask(
                        attention_maps=attention_maps_list, mask_indices=mask_indices, size=(noise_preds_tensor.shape[-2], noise_preds_tensor.shape[-1]),
                        mask_thresholding_alpha=mask_threshold_alpha, mask_erode=mask_erode, mask_dilate=mask_dilate, mask_opening=mask_opening, mask_closing=mask_closing,
                    ).to(device=device, dtype=noise_pred_uncond.dtype)
                else:
                    masks = torch.ones_like(noise_pred_uncond)

                noise_pred_uncond = (noise_pred_uncond * masks).sum(dim=0) / masks.sum(dim=0)
                noise_pred_text = (noise_pred_text * masks).sum(dim=0) / masks.sum(dim=0)
                
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. Post-processing
        image = self.decode_latents(latents)
        image = self.image_processor.postprocess(image, output_type="pil")

        self.maybe_free_model_hooks()

        # Returning dummy values for attention_maps and masks for API consistency
        return image, [], []