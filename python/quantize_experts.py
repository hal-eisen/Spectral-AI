"""Custom 4-bit quantization for Gemma4TextExperts (raw nn.Parameter blobs).

bnb's auto-quantize only handles nn.Linear; Gemma4TextExperts stores its weights
as 3D nn.Parameter tensors and dispatches per-expert via F.linear(weight[expert_id]).
This module quantizes each expert's gate_up_proj and down_proj rows to nf4 on GPU
and patches the experts.forward to dequantize on demand.

Memory: at 4-bit, Gemma 4's full expert tensor pool drops from 42.5 GB → 11.0 GB.
With dense parts (~2 GB) + activations (~2 GB) + KV cache, the 26B model now fits
on a 16 GB GPU at ~15 GB usage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _load_expert_tensors_from_safetensors(model_dir: str, layer_idx: int):
    """Read this layer's expert tensors directly from the safetensors index.
    Returns (gate_up_proj, down_proj) as CPU bf16 tensors."""
    import json
    from pathlib import Path
    from safetensors import safe_open

    mdir = Path(model_dir)
    idx = json.loads((mdir / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    gup_key = f"model.language_model.layers.{layer_idx}.experts.gate_up_proj"
    dwn_key = f"model.language_model.layers.{layer_idx}.experts.down_proj"
    gup_file = wmap[gup_key]
    dwn_file = wmap[dwn_key]

    out = {}
    for fname, key in [(gup_file, gup_key), (dwn_file, dwn_key)]:
        with safe_open(mdir / fname, framework="pt") as st:
            out[key] = st.get_tensor(key).to(torch.bfloat16)
    return out[gup_key], out[dwn_key]


def quantize_experts_to_4bit(model, *, model_dir: str,
                              blocksize: int = 128,
                              quant_type: str = "nf4",
                              compress_statistics: bool = True,
                              max_gpu_layers: Optional[int] = None) -> int:
    """Walk Gemma4TextExperts modules; replace their float weights with 4-bit
    bnb-quantized buffers + custom forward. Returns number of experts modules
    converted.

    If max_gpu_layers is set and there are more MoE layers than that, the
    later layers' experts stay on CPU (still in bf16) and dispatched per
    forward — adds PCIe overhead for those layers but lets the model fit
    when 4-bit-on-GPU alone would OOM.
    """
    import bitsandbytes as bnb
    from bitsandbytes.functional import quantize_4bit, dequantize_4bit

    n_converted = 0
    layers = model.model.language_model.layers
    dev = torch.device("cuda")

    for li, layer in enumerate(layers):
        if not hasattr(layer, "experts"):
            continue
        if max_gpu_layers is not None and li >= max_gpu_layers:
            # Leave remaining layers' experts on CPU (their original device).
            # Restore from safetensors so they're not on `meta`.
            gup_cpu, dwn_cpu = _load_expert_tensors_from_safetensors(model_dir, li)
            layer.experts.gate_up_proj = nn.Parameter(gup_cpu, requires_grad=False)
            layer.experts.down_proj = nn.Parameter(dwn_cpu, requires_grad=False)
            continue
        experts = layer.experts
        # Read expert tensors from safetensors (the in-memory ones are on meta
        # because accelerate offloaded them). Stream into GPU one expert at a
        # time to keep peak VRAM low (the layer-aggregate would be ~1.5 GB on
        # GPU at bf16 — too much temporary).
        gup_cpu, dwn_cpu = _load_expert_tensors_from_safetensors(model_dir, li)
        n_experts = gup_cpu.shape[0]

        gup_q, gup_states, dwn_q, dwn_states = [], [], [], []
        for e in range(n_experts):
            gup_e = gup_cpu[e].to(dev).contiguous()
            dwn_e = dwn_cpu[e].to(dev).contiguous()
            packed, state = quantize_4bit(
                gup_e, blocksize=blocksize, quant_type=quant_type,
                compress_statistics=compress_statistics,
            )
            gup_q.append(packed); gup_states.append(state)
            packed, state = quantize_4bit(
                dwn_e, blocksize=blocksize, quant_type=quant_type,
                compress_statistics=compress_statistics,
            )
            dwn_q.append(packed); dwn_states.append(state)
            del gup_e, dwn_e  # release the bf16 staging tensor

        # Free originals
        del experts.gate_up_proj
        del experts.down_proj
        experts.gate_up_proj = None
        experts.down_proj = None
        del gup_cpu, dwn_cpu
        torch.cuda.empty_cache()

        # Stash quantized state on the experts module
        experts._q_gate_up_packed = gup_q
        experts._q_gate_up_states = gup_states
        experts._q_down_packed = dwn_q
        experts._q_down_states = dwn_states
        experts._q_blocksize = blocksize
        experts._q_quant_type = quant_type

        # Patch forward
        experts.forward = _make_4bit_experts_forward(experts)
        n_converted += 1

        if (li + 1) % 5 == 0 or li == len(layers) - 1:
            free, total = torch.cuda.mem_get_info()
            print(f"  layer {li:>2}: experts converted to 4bit  "
                  f"VRAM used={total/1024**3 - free/1024**3:.2f}GB",
                  flush=True)

    return n_converted


def _make_4bit_experts_forward(experts):
    """Build a forward closure that mirrors Gemma4TextExperts.forward but
    dequantizes the touched expert's gate_up_proj/down_proj on demand."""
    from bitsandbytes.functional import dequantize_4bit

    n_experts = experts.num_experts
    act_fn = experts.act_fn

    @torch.no_grad()
    def forward(hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        # Same dispatch logic as the original — but dequantize per-expert before linear
        expert_mask = F.one_hot(top_k_index, num_classes=n_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            e = expert_idx[0]
            if e == n_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current_state = hidden_states[token_idx]
            # Dequantize on demand
            gup_w = dequantize_4bit(
                experts._q_gate_up_packed[e], experts._q_gate_up_states[e]
            )
            dwn_w = dequantize_4bit(
                experts._q_down_packed[e], experts._q_down_states[e]
            )
            gate, up = F.linear(current_state, gup_w).chunk(2, dim=-1)
            current_hidden = act_fn(gate) * up
            current_hidden = F.linear(current_hidden, dwn_w)
            current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current_hidden.to(final.dtype))

        return final

    return forward
