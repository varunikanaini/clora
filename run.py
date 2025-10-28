import argparse
import torch
from pathlib import Path
import sys
from diffusers import QwenImagePipeline, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, Qwen2_5_VLModel as TextEncoder
from safetensors.torch import load_file
from pytorch_metric_learning import losses

# Import our verified utilities
from utils_qwen import QwenAttentionStore, register_attention_control

# --- Helper Functions ---
def load_pipeline_with_custom_assets(args):
    pipe = QwenImagePipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    text_encoder = TextEncoder.from_pretrained(args.model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16).to("cuda")
    text_encoder.resize_token_embeddings(len(tokenizer))
    main_state_dict = text_encoder.language_model.state_dict()
    custom_embeddings = load_file(args.embeddings_path)["emb_params"].to("cuda")
    offset = len(tokenizer.get_vocab()) - custom_embeddings.size(0)
    main_state_dict["embed_tokens.weight"][offset:] = custom_embeddings
    text_encoder.language_model.load_state_dict(main_state_dict)
    pipe.tokenizer, pipe.text_encoder = tokenizer, text_encoder
    print("Custom assets loaded successfully.")
    return pipe

def get_token_indices(tokenizer, prompt, concept_str):
    token_ids = tokenizer.encode(prompt)
    tokens = [tokenizer.decode([t]) for t in token_ids]
    for start_idx in range(len(tokens)):
        for end_idx in range(start_idx + 1, len(tokens) + 1):
            current_str = "".join(tokens[start_idx:end_idx]).replace(' ', ' ').strip()
            if current_str == concept_str:
                return list(range(start_idx, end_idx))
    print(f"Warning: Concept '{concept_str}' not found in prompt '{prompt}'", file=sys.stderr)
    return []

def calculate_contrastive_loss(attention_maps: list[torch.Tensor], concept_groups: list[list[list[int]]], temperature: float = 0.5):
    embeddings = []
    labels = []
    print("\n--- Preparing Embeddings for Loss Calculation ---")
    for concept_id, group in enumerate(concept_groups):
        for prompt_id, indices in enumerate(group):
            if not indices: continue
            attn_map = attention_maps[prompt_id]
            concept_attn = attn_map[:, indices].mean(dim=1)
            concept_attn = (concept_attn - concept_attn.min()) / (concept_attn.max() - concept_attn.min() + 1e-6)
            embeddings.append(concept_attn)
            labels.append(concept_id)
            print(f"  - Added embedding for Concept {concept_id} from Prompt {prompt_id}")
    if not embeddings:
        print("No valid embeddings found to calculate loss.")
        return torch.tensor(0.0)
    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels, device=embeddings.device)
    loss_func = losses.NTXentLoss(temperature=temperature)
    loss = loss_func(embeddings, labels)
    return loss

def main():
    parser = argparse.ArgumentParser(description="Step 1: Generate Attention Maps and Calculate Contrastive Loss.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--lora_paths", nargs='+', required=True)
    parser.add_argument("--lora_trigger_tokens", nargs='+', required=True)
    parser.add_argument("--lora_generic_concepts", nargs='+', required=True)
    parser.add_argument("--prompt_template", type=str, default="a photo of [C1] and [C2]")
    args = parser.parse_args()

    pipe = load_pipeline_with_custom_assets(args)
    lora_names = [f"lora_{i}" for i in range(len(args.lora_paths))]
    for name, path in zip(lora_names, args.lora_paths):
        pipe.load_lora_weights(path, adapter_name=name)

    prompts = []
    p0 = args.prompt_template.replace("[C1]", args.lora_generic_concepts[0]).replace("[C2]", args.lora_generic_concepts[1])
    prompts.append(p0)
    p1 = args.prompt_template.replace("[C1]", args.lora_trigger_tokens[0]).replace("[C2]", args.lora_generic_concepts[1])
    prompts.append(p1)
    p2 = args.prompt_template.replace("[C1]", args.lora_generic_concepts[0]).replace("[C2]", args.lora_trigger_tokens[1])
    prompts.append(p2)
    lora_activation_list = ["", lora_names[0], lora_names[1]]
    print("\n--- Step-by-Step Prompts ---")
    print(f"  Prompt 0 (No LoRA): '{prompts[0]}'")
    print(f"  Prompt 1 (LoRA 0):  '{prompts[1]}'")
    print(f"  Prompt 2 (LoRA 1):  '{prompts[2]}'")

    c1_group = [
        get_token_indices(pipe.tokenizer, prompts[0], args.lora_generic_concepts[0]),
        get_token_indices(pipe.tokenizer, prompts[1], args.lora_trigger_tokens[0]),
        get_token_indices(pipe.tokenizer, prompts[2], args.lora_generic_concepts[0]),
    ]
    c2_group = [
        get_token_indices(pipe.tokenizer, prompts[0], args.lora_generic_concepts[1]),
        get_token_indices(pipe.tokenizer, prompts[1], args.lora_generic_concepts[1]),
        get_token_indices(pipe.tokenizer, prompts[2], args.lora_trigger_tokens[1]),
    ]
    concept_groups = [c1_group, c2_group]
    print("\n--- Concept Groups for Loss ---")
    print(f"  Concept 1 ('{args.lora_generic_concepts[0]}'): {c1_group}")
    print(f"  Concept 2 ('{args.lora_generic_concepts[1]}'): {c2_group}")

    height, width = 1024, 1024
    model_dtype = pipe.transformer.dtype

    generator = torch.Generator(device="cuda").manual_seed(42)
    latents = pipe.prepare_latents(1, pipe.vae.config.z_dim, height, width, dtype=model_dtype, device="cuda", generator=generator)

    timestep = torch.tensor([999.0], dtype=model_dtype, device="cuda")
    img_shapes = [(1, height // 16, width // 16)]
    attention_store = QwenAttentionStore(attn_res=(height // 16, width // 16))
    register_attention_control(pipe, attention_store)

    collected_attention_maps = []
    print("\n--- Performing Forward Passes to Generate Attention Maps ---")
    for i, (lora_name, prompt) in enumerate(zip(lora_activation_list, prompts)):
        attention_store.reset()
        prompt_embeds, prompt_mask = pipe.encode_prompt(prompt, device="cuda")
        if lora_name: pipe.set_adapters([lora_name])
        else: pipe.disable_lora()
        txt_seq_lens = [prompt_embeds.shape[1]]
        with torch.no_grad():
             _ = pipe.transformer(
                latents,
                timestep=timestep / 1000,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens
            )
        agg_map = attention_store.aggregate_attention()
        collected_attention_maps.append(agg_map)
        print(f"  - Pass {i}: Captured attention map of shape {agg_map.shape}")

    final_loss = calculate_contrastive_loss(collected_attention_maps, concept_groups)
    
    print("\n" + "="*50)
    print(f"✅ FINAL CONTRASTIVE LOSS: {final_loss.item()}")
    print("="*50)
    print("\nStep 1 complete. If the loss is a valid number, the core logic is correct.")

if __name__ == "__main__":
    main()