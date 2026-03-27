"""
Hybrid talker decoder for Qwen3-TTS.

Two phases of generation:

  prefill(input_embeds)   — PyTorch forward pass over the full text input.
                            Populates the KV cache. Returns first audio token.

  decode_step(token_id)   — Single-step decode, one audio token at a time.
                            PyTorch fallback locally; megakernel swapped in on GPU (Step 15).

After prefill, call sync_kv_to_megakernel(decoder) to copy the KV cache into
the megakernel Decoder's buffers before starting the fast decode loop.
"""

import torch
import torch.nn.functional as F

NUM_LAYERS    = 28
NUM_Q_HEADS   = 16
NUM_KV_HEADS  = 8
HEAD_DIM      = 128
HIDDEN_SIZE   = 1024   # NUM_Q_HEADS * HEAD_DIM
Q_SIZE        = NUM_Q_HEADS * HEAD_DIM   # 2048
KV_SIZE       = NUM_KV_HEADS * HEAD_DIM  # 1024
INTERMEDIATE  = 3072
MAX_SEQ_LEN   = 2048
ROPE_THETA    = 1_000_000
RMS_EPS       = 1e-6
GQA_GROUPS    = NUM_Q_HEADS // NUM_KV_HEADS  # 2


class MegakernelTalker:
    """
    Hybrid prefill (PyTorch) + decode (PyTorch fallback / megakernel on GPU).
    """

    def __init__(self, weights: dict, device: str = "cpu"):
        self.device = device
        self.w = weights
        self._position = 0

        # RoPE inverse frequencies — same theta as weight bridge
        self._inv_freq = 1.0 / (
            ROPE_THETA ** (
                torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM
            )
        )

        # KV cache layout matches megakernel Decoder exactly so sync is a direct copy
        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device=device,
        )
        self._v_cache = torch.zeros_like(self._k_cache)

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prefill(self, input_embeds: torch.Tensor) -> int:
        """
        Process projected text embeddings and return the first audio token ID.

        Args:
            input_embeds: [seq_len, HIDDEN_SIZE] float32 or bfloat16
                          caller is responsible for: text_tokens
                          → text_embedding [151936,2048]
                          → text_projection (2048→1024)
                          → this tensor
        Returns:
            first audio token id  (int, 0–3071)
        """
        seq_len = input_embeds.shape[0]
        hidden = input_embeds.to(torch.bfloat16).to(self.device)
        positions = torch.arange(seq_len, device=self.device)

        hidden = self._forward_layers(hidden, positions, start_pos=0)

        # Take the last token's representation
        last = self._rms_norm(hidden[-1:], self.w["final_norm_weight"])
        logits = last @ self.w["lm_head_weight"].T   # [1, 3072]

        self._position = seq_len
        self._last_hidden = last  # [1, 1024] — needed by code_predictor
        return int(logits[0].argmax())

    def decode_step(self, token_id: int) -> int:
        """
        Single audio-token decode step — PyTorch fallback.
        On the GPU machine this path is replaced by megakernel.step() (Step 15).

        Args:
            token_id: previous audio token (0–3071)
        Returns:
            next audio token id
        """
        # Look up the audio token embedding  [1, 1024]
        hidden = self.w["embed_weight"][token_id].unsqueeze(0).to(self.device)
        positions = torch.tensor([self._position], device=self.device)

        hidden = self._forward_layers(hidden, positions, start_pos=self._position)

        hidden = self._rms_norm(hidden, self.w["final_norm_weight"])
        logits = hidden @ self.w["lm_head_weight"].T   # [1, 3072]

        self._last_hidden = hidden  # [1, 1024] — needed by code_predictor
        self._position += 1
        return int(logits[0].argmax())

    def decode_step_from_embed(self, input_embed: torch.Tensor) -> int:
        """
        Decode step using a pre-computed embedding instead of a token id.
        Required for multi-codebook TTS where the next input is the
        sum of all 16 code embeddings (semantic + 15 acoustic).

        Args:
            input_embed: [1, HIDDEN_SIZE] bfloat16 combined embedding
        Returns:
            next semantic token id
        """
        hidden = input_embed.to(torch.bfloat16).to(self.device)
        positions = torch.tensor([self._position], device=self.device)
        hidden = self._forward_layers(hidden, positions, start_pos=self._position)
        hidden = self._rms_norm(hidden, self.w["final_norm_weight"])
        logits = hidden @ self.w["lm_head_weight"].T   # [1, 3072]
        self._last_hidden = hidden
        self._position += 1
        return int(logits[0].argmax())

    def sync_kv_to_megakernel(self, decoder) -> None:
        """
        Copy prefill KV cache into megakernel Decoder's buffers.
        Must be called after prefill() and before the first decoder.step() call.

        Args:
            decoder: qwen_megakernel.Decoder instance
        """
        n = self._position
        decoder._k_cache[:, :, :n, :] = self._k_cache[:, :, :n, :]
        decoder._v_cache[:, :, :n, :] = self._v_cache[:, :, :n, :]
        decoder._position = n

    # ------------------------------------------------------------------
    # Transformer internals
    # ------------------------------------------------------------------

    def _forward_layers(
        self,
        hidden: torch.Tensor,     # [seq, HIDDEN_SIZE]
        positions: torch.Tensor,  # [seq]
        start_pos: int,
    ) -> torch.Tensor:
        for i in range(NUM_LAYERS):
            hidden = self._layer(hidden, positions, layer_idx=i, start_pos=start_pos)
        return hidden

    def _layer(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        layer_idx: int,
        start_pos: int,
    ) -> torch.Tensor:
        lw = self._layer_weights(layer_idx)
        seq = hidden.shape[0]
        end_pos = start_pos + seq

        # --- Self-attention with pre-norm ---
        residual = hidden
        normed = self._rms_norm(hidden, lw["input_layernorm"])

        q = normed @ lw["q_proj"].T                      # [seq, Q_SIZE=2048]
        k = normed @ lw["k_proj"].T                      # [seq, KV_SIZE=1024]
        v = normed @ lw["v_proj"].T                      # [seq, KV_SIZE=1024]

        q = q.view(seq, NUM_Q_HEADS, HEAD_DIM)
        k = k.view(seq, NUM_KV_HEADS, HEAD_DIM)
        v = v.view(seq, NUM_KV_HEADS, HEAD_DIM)

        # Per-head QK norm (Qwen3 uses this instead of attention scaling alone)
        q = self._rms_norm(q, lw["q_norm"])
        k = self._rms_norm(k, lw["k_norm"])

        # Rotary position embeddings
        q, k = self._apply_rope(q, k, positions)

        # Write new K,V into cache   [layer, kv_heads, seq_pos, head_dim]
        self._k_cache[layer_idx, :, start_pos:end_pos, :] = k.transpose(0, 1)
        self._v_cache[layer_idx, :, start_pos:end_pos, :] = v.transpose(0, 1)

        # Attend over full context so far
        k_full = self._k_cache[layer_idx, :, :end_pos, :]  # [kv_heads, end_pos, head_dim]
        v_full = self._v_cache[layer_idx, :, :end_pos, :]

        attn_out = self._gqa_attention(q, k_full, v_full, start_pos)  # [seq, Q_SIZE]
        hidden = residual + attn_out @ lw["o_proj"].T                  # [seq, HIDDEN_SIZE]

        # --- MLP with pre-norm ---
        residual = hidden
        normed = self._rms_norm(hidden, lw["post_attn_layernorm"])

        gate = F.silu(normed @ lw["gate_proj"].T)   # [seq, INTERMEDIATE]
        up   = normed @ lw["up_proj"].T              # [seq, INTERMEDIATE]
        hidden = (gate * up) @ lw["down_proj"].T     # [seq, HIDDEN_SIZE]

        return residual + hidden

    def _gqa_attention(
        self,
        q: torch.Tensor,   # [seq, num_q_heads, head_dim]
        k: torch.Tensor,   # [num_kv_heads, cache_len, head_dim]
        v: torch.Tensor,   # [num_kv_heads, cache_len, head_dim]
        start_pos: int,
    ) -> torch.Tensor:
        seq       = q.shape[0]
        cache_len = k.shape[1]

        # Reshape Q for grouped-query attention
        # [num_kv_heads, gqa_groups, seq, head_dim]
        q = q.transpose(0, 1).view(NUM_KV_HEADS, GQA_GROUPS, seq, HEAD_DIM)
        k = k.unsqueeze(1)  # [num_kv_heads, 1, cache_len, head_dim]
        v = v.unsqueeze(1)

        scale = HEAD_DIM ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [kv_heads, groups, seq, cache_len]

        # Causal mask: token at (start_pos + i) cannot attend to positions after it
        if seq > 1:
            q_pos = torch.arange(start_pos, start_pos + seq, device=q.device).unsqueeze(1)
            k_pos = torch.arange(cache_len, device=k.device).unsqueeze(0)
            mask  = (k_pos > q_pos).unsqueeze(0).unsqueeze(0)  # [1, 1, seq, cache_len]
            attn  = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out  = torch.matmul(attn, v)  # [kv_heads, groups, seq, head_dim]

        # Merge back to [seq, Q_SIZE]
        out = out.permute(2, 0, 1, 3).reshape(seq, Q_SIZE)
        return out

    def _apply_rope(
        self,
        q: torch.Tensor,  # [seq, num_q_heads, head_dim]
        k: torch.Tensor,  # [seq, num_kv_heads, head_dim]
        positions: torch.Tensor,  # [seq]
    ):
        freqs = torch.outer(positions.float(), self._inv_freq)  # [seq, head_dim/2]
        cos   = torch.cos(freqs).repeat(1, 2).to(q.dtype)       # [seq, head_dim]
        sin   = torch.sin(freqs).repeat(1, 2).to(q.dtype)

        def rotate(x):
            # x: [seq, heads, head_dim]
            cos_ = cos.unsqueeze(1)   # [seq, 1, head_dim]
            sin_ = sin.unsqueeze(1)
            half = x.shape[-1] // 2
            x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
            return x * cos_ + x_rot * sin_

        return rotate(q), rotate(k)

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + RMS_EPS)
        return (weight.float() * x).to(orig_dtype)

    def _layer_weights(self, i: int) -> dict:
        """Unpack the flat layer_weights list into a named dict for layer i."""
        base = i * 11
        lw   = self.w["layer_weights"]
        return {
            "input_layernorm":     lw[base + 0],
            "q_proj":              lw[base + 1],
            "k_proj":              lw[base + 2],
            "v_proj":              lw[base + 3],
            "q_norm":              lw[base + 4],
            "k_norm":              lw[base + 5],
            "o_proj":              lw[base + 6],
            "post_attn_layernorm": lw[base + 7],
            "gate_proj":           lw[base + 8],
            "up_proj":             lw[base + 9],
            "down_proj":           lw[base + 10],
        }
