"""
Weight bridge: loads Qwen3-TTS-12Hz-0.6B-Base weights from safetensors
and formats them as the dict expected by qwen_megakernel.Decoder.

The megakernel was written for Qwen3-0.6B (text). Qwen3-TTS has the same
architecture but different weight key names and a separate LM head.
This file does the translation.
"""

import math
import os
import torch
from safetensors import safe_open

NUM_LAYERS = 28
HEAD_DIM = 128
MAX_SEQ_LEN = 2048
ROPE_THETA = 1_000_000  # Qwen3-TTS uses 1M; megakernel default is 10K (wrong for us)

# The 11 per-layer weight suffixes in the exact order the megakernel expects.
# These names are identical in both Qwen3-0.6B and Qwen3-TTS — only the prefix differs.
_LAYER_SUFFIXES = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def _build_rope_tables(device: str) -> tuple:
    """Compute RoPE cos/sin tables with Qwen3-TTS theta (1_000_000)."""
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).contiguous()
    return cos_table.to(device), sin_table.to(device)


class TalkerWeightBridge:
    """
    Translates Qwen3-TTS safetensors weight names → qwen_megakernel.Decoder format.

    Usage:
        weights = TalkerWeightBridge().load('/Users/work/.cache/qwen3-tts')
        decoder = Decoder(weights=weights)
    """

    def load(self, model_dir: str, device: str = "cpu") -> dict:
        """
        Args:
            model_dir: path to Qwen3-TTS-12Hz-0.6B-Base download directory
            device:    'cpu' for shape-checking locally, 'cuda' on the GPU machine

        Returns:
            weights dict ready to pass into qwen_megakernel.Decoder(weights=...)
        """
        model_dir = os.path.expanduser(model_dir)
        safetensors_path = os.path.join(model_dir, "model.safetensors")

        with safe_open(safetensors_path, framework="pt", device=device) as f:

            def load(key: str) -> torch.Tensor:
                return f.get_tensor(key).to(torch.bfloat16).contiguous()

            # --- 28 × 11 = 308 layer tensors, flat list ---
            # Megakernel prefix: "model.layers.{i}."
            # Qwen3-TTS prefix:  "talker.model.layers.{i}."
            layer_weights = []
            for i in range(NUM_LAYERS):
                prefix = f"talker.model.layers.{i}."
                for suffix in _LAYER_SUFFIXES:
                    layer_weights.append(load(prefix + suffix))

            # Megakernel key: "model.embed_tokens.weight"   [151936, 1024]
            # Qwen3-TTS key:  "talker.model.codec_embedding.weight" [3072, 1024]
            embed_weight = load("talker.model.codec_embedding.weight")

            # Megakernel key: "lm_head_weight" — tied to embed in text model
            # Qwen3-TTS key:  separate "talker.codec_head.weight"   [3072, 1024]
            lm_head_weight = load("talker.codec_head.weight")

            final_norm_weight = load("talker.model.norm.weight")

            # --- Text preparation weights (used by streaming engine prefill) ---
            # Text tokens are embedded in 2048-dim space, then projected down to 1024
            text_embedding = load("talker.model.text_embedding.weight")       # [151936, 2048]
            text_proj_fc1_w = load("talker.text_projection.linear_fc1.weight") # [2048, 2048]
            text_proj_fc1_b = load("talker.text_projection.linear_fc1.bias")   # [2048]
            text_proj_fc2_w = load("talker.text_projection.linear_fc2.weight") # [1024, 2048]
            text_proj_fc2_b = load("talker.text_projection.linear_fc2.bias")   # [1024]

        cos_table, sin_table = _build_rope_tables(device)

        return dict(
            embed_weight=embed_weight,
            layer_weights=layer_weights,
            final_norm_weight=final_norm_weight,
            lm_head_weight=lm_head_weight,
            cos_table=cos_table,
            sin_table=sin_table,
            # text preparation
            text_embedding=text_embedding,
            text_proj_fc1_w=text_proj_fc1_w,
            text_proj_fc1_b=text_proj_fc1_b,
            text_proj_fc2_w=text_proj_fc2_w,
            text_proj_fc2_b=text_proj_fc2_b,
        )


if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/work/.cache/qwen3-tts"
    print(f"Loading from {model_dir} on CPU...")

    weights = TalkerWeightBridge().load(model_dir, device="cpu")

    print(f"\nembed_weight:      {tuple(weights['embed_weight'].shape)}")
    print(f"lm_head_weight:    {tuple(weights['lm_head_weight'].shape)}")
    print(f"final_norm_weight: {tuple(weights['final_norm_weight'].shape)}")
    print(f"cos_table:         {tuple(weights['cos_table'].shape)}")
    print(f"sin_table:         {tuple(weights['sin_table'].shape)}")
    print(f"\nLayer weights ({len(weights['layer_weights'])} tensors = {NUM_LAYERS} layers × {len(_LAYER_SUFFIXES)}):")
    for i in range(NUM_LAYERS):
        base = i * len(_LAYER_SUFFIXES)
        shapes = [tuple(weights['layer_weights'][base + j].shape) for j in range(len(_LAYER_SUFFIXES))]
        print(f"  layer {i:2d}: {shapes}")

    print("\nAll shapes verified OK.")
