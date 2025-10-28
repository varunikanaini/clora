

import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from collections import defaultdict
from tqdm import tqdm

# --- Step 1: Initialize the Pipeline ---
pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype=torch.bfloat16,
).to("cuda")

# --- Step 2: Load the LoRA Weights ---
lora_path = "/workspace/shivanvitha/ai-toolkit/output/satyadev_man/satyadev_man_LoRA_000004263_best.safetensors"
lora_state_dict = load_file(lora_path)

# --- Step 3: Manual LoRA Weight Merging ---
LORA_SCALE = 1.0

# Get the base model's state dict
transformer_state_dict = pipeline.transformer.state_dict()

lora_layers = defaultdict(dict)
for key, value in lora_state_dict.items():
    base_key = key.replace("diffusion_model.", "").replace(".lora_A.weight", "").replace(".lora_B.weight", "")
    
    if "lora_A" in key:
        lora_layers[base_key]['lora_A.weight'] = value
    elif "lora_B" in key:
        lora_layers[base_key]['lora_B.weight'] = value

print(f"Found {len(lora_layers)} LoRA-modified layers to merge.")

# Merge each LoRA layer into the base model
for base_key, loras in tqdm(lora_layers.items(), desc="Merging LoRA weights"):
    target_weight_key = f"{base_key}.weight"
    
    if target_weight_key in transformer_state_dict:
        lora_A = loras['lora_A.weight'].to(dtype=torch.bfloat16, device="cuda")
        lora_B = loras['lora_B.weight'].to(dtype=torch.bfloat16, device="cuda")
        
        lora_update = lora_B @ lora_A

        original_weight = transformer_state_dict[target_weight_key].to(device="cuda")
        transformer_state_dict[target_weight_key] = original_weight + (lora_update * LORA_SCALE)
    else:
        print(f"Warning: Target weight {target_weight_key} not found in model state_dict. Skipping.")

# Load the updated state dict back into the transformer
pipeline.transformer.load_state_dict(transformer_state_dict)
print("LoRA weights successfully merged into the model.")

prompt = "a portrait of satyadev_man in a futuristic city"
negative_prompt = "nsfw, blurry"
height = 512
width = 512
num_inference_steps = 50
guidance_scale = 4.0
seed = 53

image = pipeline(
    prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=torch.Generator(device=pipeline.device).manual_seed(seed),
).images[0]

# --- Step 5: Save the Result ---
image.save("qwen_image_satyadev_man_merged.png")
print("Image saved successfully!")