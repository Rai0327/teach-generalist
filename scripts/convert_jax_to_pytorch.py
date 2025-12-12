#!/usr/bin/env python3
"""Convert JAX pi0/pi05 checkpoint to PyTorch safetensors format.

Usage:
    python scripts/convert_jax_to_pytorch.py \
        --jax-checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
        --output-path checkpoints/pi05_libero_pytorch/model.safetensors
"""

import argparse
import gc
import logging
import os
import pathlib
import sys

import numpy as np
import torch
import safetensors.torch
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from openpi.models import model as _model


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict to flat dict with dotted keys."""
    flat = {}
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, new_key))
        else:
            flat[new_key] = np.array(value) if hasattr(value, "__array__") else value
    return flat


def detect_model_type(flat_jax: dict) -> tuple[bool, int, int]:
    """Detect if this is pi0 or pi05, and number of layers."""
    is_pi05 = "time_mlp_in.kernel" in flat_jax
    
    if "PaliGemma.img.Transformer.encoderblock.LayerNorm_0.scale" in flat_jax:
        num_vision_layers = flat_jax["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.scale"].shape[0]
    else:
        num_vision_layers = 27
    
    if "PaliGemma.llm.layers.pre_attention_norm.scale" in flat_jax:
        num_llm_layers = flat_jax["PaliGemma.llm.layers.pre_attention_norm.scale"].shape[0]
    else:
        num_llm_layers = 18
    
    return is_pi05, num_vision_layers, num_llm_layers


def to_torch(jax_array: np.ndarray) -> torch.Tensor:
    """Convert JAX/numpy array to PyTorch tensor, handling bfloat16."""
    if jax_array.dtype == np.dtype('bfloat16') or str(jax_array.dtype) == 'bfloat16':
        arr = jax_array.astype(np.float32)
        return torch.from_numpy(arr.copy()).to(torch.bfloat16).contiguous()
    return torch.from_numpy(jax_array.copy()).contiguous()


def convert_and_add(converted: dict, pt_key: str, jax_array: np.ndarray, transpose: bool = False):
    """Convert a single parameter and add to dict."""
    tensor = to_torch(jax_array)
    if transpose and tensor.ndim == 2:
        tensor = tensor.T.contiguous()
    converted[pt_key] = tensor.contiguous()


def convert_checkpoint(jax_checkpoint_path: str, output_path: str):
    """Convert JAX checkpoint to PyTorch safetensors format."""
    
    logger.info(f"Loading JAX checkpoint from: {jax_checkpoint_path}")
    logger.info("Loading with bfloat16 to reduce memory...")
    jax_params = _model.restore_params(
        pathlib.Path(jax_checkpoint_path) / "params",
        restore_type=np.ndarray,
        dtype=jnp.bfloat16,
    )
    flat_jax = flatten_dict(jax_params)
    del jax_params
    gc.collect()
    logger.info(f"JAX checkpoint loaded: {len(flat_jax)} parameters")
    
    is_pi05, num_vision_layers, num_llm_layers = detect_model_type(flat_jax)
    logger.info(f"Detected: pi05={is_pi05}, vision_layers={num_vision_layers}, llm_layers={num_llm_layers}")
    
    converted = {}
    
    # =====================================================================
    # 1. Simple 1:1 mappings
    # =====================================================================
    logger.info("Converting action projections...")
    
    simple_mappings = [
        ("action_in_proj.weight", "action_in_proj.kernel", True),
        ("action_in_proj.bias", "action_in_proj.bias", False),
        ("action_out_proj.weight", "action_out_proj.kernel", True),
        ("action_out_proj.bias", "action_out_proj.bias", False),
    ]
    
    if is_pi05:
        simple_mappings.extend([
            ("time_mlp_in.weight", "time_mlp_in.kernel", True),
            ("time_mlp_in.bias", "time_mlp_in.bias", False),
            ("time_mlp_out.weight", "time_mlp_out.kernel", True),
            ("time_mlp_out.bias", "time_mlp_out.bias", False),
        ])
    else:
        simple_mappings.extend([
            ("state_proj.weight", "state_proj.kernel", True),
            ("state_proj.bias", "state_proj.bias", False),
            ("action_time_mlp_in.weight", "action_time_mlp_in.kernel", True),
            ("action_time_mlp_in.bias", "action_time_mlp_in.bias", False),
            ("action_time_mlp_out.weight", "action_time_mlp_out.kernel", True),
            ("action_time_mlp_out.bias", "action_time_mlp_out.bias", False),
        ])
    
    for pt_key, jax_key, transpose in simple_mappings:
        if jax_key in flat_jax:
            convert_and_add(converted, pt_key, flat_jax[jax_key], transpose)
    
    # =====================================================================
    # 2. Vision encoder
    # =====================================================================
    logger.info("Converting vision encoder...")
    
    if "PaliGemma.img.embedding.kernel" in flat_jax:
        kernel = flat_jax["PaliGemma.img.embedding.kernel"]
        tensor = to_torch(kernel).permute(3, 2, 0, 1).contiguous()
        converted["paligemma_with_expert.paligemma.vision_tower.vision_model.embeddings.patch_embedding.weight"] = tensor
    
    if "PaliGemma.img.embedding.bias" in flat_jax:
        convert_and_add(converted, "paligemma_with_expert.paligemma.vision_tower.vision_model.embeddings.patch_embedding.bias",
                       flat_jax["PaliGemma.img.embedding.bias"])
    
    if "PaliGemma.img.pos_embedding" in flat_jax:
        pos_emb = flat_jax["PaliGemma.img.pos_embedding"][0]
        convert_and_add(converted, "paligemma_with_expert.paligemma.vision_tower.vision_model.embeddings.position_embedding.weight", pos_emb)
    
    # Vision blocks
    for layer_idx in range(num_vision_layers):
        prefix = f"paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers.{layer_idx}"
        
        if "PaliGemma.img.Transformer.encoderblock.LayerNorm_0.scale" in flat_jax:
            convert_and_add(converted, f"{prefix}.layer_norm1.weight",
                           flat_jax["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.scale"][layer_idx])
            convert_and_add(converted, f"{prefix}.layer_norm1.bias",
                           flat_jax["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.bias"][layer_idx])
            convert_and_add(converted, f"{prefix}.layer_norm2.weight",
                           flat_jax["PaliGemma.img.Transformer.encoderblock.LayerNorm_1.scale"][layer_idx])
            convert_and_add(converted, f"{prefix}.layer_norm2.bias",
                           flat_jax["PaliGemma.img.Transformer.encoderblock.LayerNorm_1.bias"][layer_idx])
        
        if "PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.kernel" in flat_jax:
            convert_and_add(converted, f"{prefix}.mlp.fc1.weight",
                           flat_jax["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.kernel"][layer_idx], True)
            convert_and_add(converted, f"{prefix}.mlp.fc1.bias",
                           flat_jax["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.bias"][layer_idx])
            convert_and_add(converted, f"{prefix}.mlp.fc2.weight",
                           flat_jax["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.kernel"][layer_idx], True)
            convert_and_add(converted, f"{prefix}.mlp.fc2.bias",
                           flat_jax["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.bias"][layer_idx])
        
        if "PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.kernel" in flat_jax:
            q_kernel = flat_jax["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.kernel"][layer_idx]
            q_bias = flat_jax["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.bias"][layer_idx]
            k_kernel = flat_jax["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.key.kernel"][layer_idx]
            k_bias = flat_jax["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.key.bias"][layer_idx]
            v_kernel = flat_jax["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.value.kernel"][layer_idx]
            v_bias = flat_jax["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.value.bias"][layer_idx]
            o_kernel = flat_jax["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.out.kernel"][layer_idx]
            o_bias = flat_jax["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.out.bias"][layer_idx]
            
            hidden_dim = q_kernel.shape[0]
            
            converted[f"{prefix}.self_attn.q_proj.weight"] = to_torch(q_kernel.reshape(hidden_dim, -1).T)
            converted[f"{prefix}.self_attn.q_proj.bias"] = to_torch(q_bias.reshape(-1))
            converted[f"{prefix}.self_attn.k_proj.weight"] = to_torch(k_kernel.reshape(hidden_dim, -1).T)
            converted[f"{prefix}.self_attn.k_proj.bias"] = to_torch(k_bias.reshape(-1))
            converted[f"{prefix}.self_attn.v_proj.weight"] = to_torch(v_kernel.reshape(hidden_dim, -1).T)
            converted[f"{prefix}.self_attn.v_proj.bias"] = to_torch(v_bias.reshape(-1))
            converted[f"{prefix}.self_attn.out_proj.weight"] = to_torch(o_kernel.reshape(-1, hidden_dim).T)
            converted[f"{prefix}.self_attn.out_proj.bias"] = to_torch(o_bias)
    
    if "PaliGemma.img.Transformer.encoder_norm.scale" in flat_jax:
        convert_and_add(converted, "paligemma_with_expert.paligemma.vision_tower.vision_model.post_layernorm.weight",
                       flat_jax["PaliGemma.img.Transformer.encoder_norm.scale"])
        convert_and_add(converted, "paligemma_with_expert.paligemma.vision_tower.vision_model.post_layernorm.bias",
                       flat_jax["PaliGemma.img.Transformer.encoder_norm.bias"])
    
    if "PaliGemma.img.head.kernel" in flat_jax:
        convert_and_add(converted, "paligemma_with_expert.paligemma.multi_modal_projector.linear.weight",
                       flat_jax["PaliGemma.img.head.kernel"], True)
        convert_and_add(converted, "paligemma_with_expert.paligemma.multi_modal_projector.linear.bias",
                       flat_jax["PaliGemma.img.head.bias"])
    
    # =====================================================================
    # 3. Language model
    # =====================================================================
    logger.info("Converting language model...")
    
    if "PaliGemma.llm.embedder.input_embedding" in flat_jax:
        convert_and_add(converted, "paligemma_with_expert.paligemma.language_model.model.embed_tokens.weight",
                       flat_jax["PaliGemma.llm.embedder.input_embedding"])
    
    if "PaliGemma.llm.final_norm.scale" in flat_jax:
        convert_and_add(converted, "paligemma_with_expert.paligemma.language_model.model.norm.weight",
                       flat_jax["PaliGemma.llm.final_norm.scale"])
    
    for layer_idx in range(num_llm_layers):
        prefix = f"paligemma_with_expert.paligemma.language_model.model.layers.{layer_idx}"
        
        if "PaliGemma.llm.layers.pre_attention_norm.scale" in flat_jax:
            convert_and_add(converted, f"{prefix}.input_layernorm.weight",
                           flat_jax["PaliGemma.llm.layers.pre_attention_norm.scale"][layer_idx])
        
        if "PaliGemma.llm.layers.pre_ffw_norm.scale" in flat_jax:
            convert_and_add(converted, f"{prefix}.post_attention_layernorm.weight",
                           flat_jax["PaliGemma.llm.layers.pre_ffw_norm.scale"][layer_idx])
        
        if "PaliGemma.llm.layers.attn.q_einsum.w" in flat_jax:
            q_w = flat_jax["PaliGemma.llm.layers.attn.q_einsum.w"][layer_idx]
            num_heads, hidden_dim, head_dim = q_w.shape
            converted[f"{prefix}.self_attn.q_proj.weight"] = to_torch(
                q_w.transpose(1, 0, 2).reshape(hidden_dim, -1).T
            )
        
        if "PaliGemma.llm.layers.attn.kv_einsum.w" in flat_jax:
            kv_w = flat_jax["PaliGemma.llm.layers.attn.kv_einsum.w"][layer_idx]
            converted[f"{prefix}.self_attn.k_proj.weight"] = to_torch(kv_w[0, 0].T)
            converted[f"{prefix}.self_attn.v_proj.weight"] = to_torch(kv_w[1, 0].T)
        
        if "PaliGemma.llm.layers.attn.attn_vec_einsum.w" in flat_jax:
            o_w = flat_jax["PaliGemma.llm.layers.attn.attn_vec_einsum.w"][layer_idx]
            converted[f"{prefix}.self_attn.o_proj.weight"] = to_torch(
                o_w.reshape(-1, o_w.shape[-1]).T
            )
        
        if "PaliGemma.llm.layers.mlp.gating_einsum" in flat_jax:
            gating = flat_jax["PaliGemma.llm.layers.mlp.gating_einsum"][layer_idx]
            converted[f"{prefix}.mlp.gate_proj.weight"] = to_torch(gating[0].T)
            converted[f"{prefix}.mlp.up_proj.weight"] = to_torch(gating[1].T)
        
        if "PaliGemma.llm.layers.mlp.linear" in flat_jax:
            converted[f"{prefix}.mlp.down_proj.weight"] = to_torch(
                flat_jax["PaliGemma.llm.layers.mlp.linear"][layer_idx].T
            )
    
    # =====================================================================
    # 4. Action expert
    # =====================================================================
    logger.info("Converting action expert...")
    
    if is_pi05:
        if "PaliGemma.llm.final_norm_1.Dense_0.kernel" in flat_jax:
            convert_and_add(converted, "paligemma_with_expert.gemma_expert.model.norm.adarms_proj.weight",
                           flat_jax["PaliGemma.llm.final_norm_1.Dense_0.kernel"], True)
            convert_and_add(converted, "paligemma_with_expert.gemma_expert.model.norm.adarms_proj.bias",
                           flat_jax["PaliGemma.llm.final_norm_1.Dense_0.bias"])
    
    for layer_idx in range(num_llm_layers):
        prefix = f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}"
        
        if is_pi05:
            if "PaliGemma.llm.layers.pre_attention_norm_1.Dense_0.kernel" in flat_jax:
                convert_and_add(converted, f"{prefix}.input_layernorm.adarms_proj.weight",
                               flat_jax["PaliGemma.llm.layers.pre_attention_norm_1.Dense_0.kernel"][layer_idx], True)
                convert_and_add(converted, f"{prefix}.input_layernorm.adarms_proj.bias",
                               flat_jax["PaliGemma.llm.layers.pre_attention_norm_1.Dense_0.bias"][layer_idx])
                convert_and_add(converted, f"{prefix}.post_attention_layernorm.adarms_proj.weight",
                               flat_jax["PaliGemma.llm.layers.pre_ffw_norm_1.Dense_0.kernel"][layer_idx], True)
                convert_and_add(converted, f"{prefix}.post_attention_layernorm.adarms_proj.bias",
                               flat_jax["PaliGemma.llm.layers.pre_ffw_norm_1.Dense_0.bias"][layer_idx])
        
        if "PaliGemma.llm.layers.attn.q_einsum_1.w" in flat_jax:
            q_w = flat_jax["PaliGemma.llm.layers.attn.q_einsum_1.w"][layer_idx]
            num_heads, hidden_dim, head_dim = q_w.shape
            converted[f"{prefix}.self_attn.q_proj.weight"] = to_torch(
                q_w.transpose(1, 0, 2).reshape(hidden_dim, -1).T
            )
        
        if "PaliGemma.llm.layers.attn.kv_einsum_1.w" in flat_jax:
            kv_w = flat_jax["PaliGemma.llm.layers.attn.kv_einsum_1.w"][layer_idx]
            converted[f"{prefix}.self_attn.k_proj.weight"] = to_torch(kv_w[0, 0].T)
            converted[f"{prefix}.self_attn.v_proj.weight"] = to_torch(kv_w[1, 0].T)
        
        if "PaliGemma.llm.layers.attn.attn_vec_einsum_1.w" in flat_jax:
            o_w = flat_jax["PaliGemma.llm.layers.attn.attn_vec_einsum_1.w"][layer_idx]
            converted[f"{prefix}.self_attn.o_proj.weight"] = to_torch(
                o_w.reshape(-1, o_w.shape[-1]).T
            )
        
        if "PaliGemma.llm.layers.mlp_1.gating_einsum" in flat_jax:
            gating = flat_jax["PaliGemma.llm.layers.mlp_1.gating_einsum"][layer_idx]
            converted[f"{prefix}.mlp.gate_proj.weight"] = to_torch(gating[0].T)
            converted[f"{prefix}.mlp.up_proj.weight"] = to_torch(gating[1].T)
        
        if "PaliGemma.llm.layers.mlp_1.linear" in flat_jax:
            converted[f"{prefix}.mlp.down_proj.weight"] = to_torch(
                flat_jax["PaliGemma.llm.layers.mlp_1.linear"][layer_idx].T
            )
    
    # Free JAX params
    del flat_jax
    gc.collect()
    
    # =====================================================================
    # 5. Save
    # =====================================================================
    logger.info(f"Converted {len(converted)} parameters")
    
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving to: {output_path}")
    safetensors.torch.save_file(converted, str(output_path))
    
    logger.info("=" * 60)
    logger.info("Conversion complete!")
    logger.info(f"Output: {output_path}")
    logger.info(f"Model type: {'pi05' if is_pi05 else 'pi0'}")
    logger.info(f"Parameters converted: {len(converted)}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("To use with train_dagger.py:")
    logger.info(f"  --pi0-checkpoint {output_path.parent}")


def main():
    parser = argparse.ArgumentParser(description="Convert JAX pi0 checkpoint to PyTorch")
    parser.add_argument("--jax-checkpoint", type=str,
                       default=os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/pi05_libero"))
    parser.add_argument("--output-path", type=str,
                       default="checkpoints/pi05_libero_pytorch/model.safetensors")
    args = parser.parse_args()
    convert_checkpoint(args.jax_checkpoint, args.output_path)


if __name__ == "__main__":
    main()
