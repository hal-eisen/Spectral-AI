"""Extract router input/output pairs for BVH router distillation.

Generic harness that works for both Gemma 4 (Gemma4TextRouter) and Qwen 3.6
(Qwen3_5MoeTopKRouter). Both expose a 3-tuple forward signature; we hook
each layer's router module, capture (router_input, top_k_index, top_k_weights),
and dump per-layer .pt files.

Usage:
    python python/extract_router_io.py \\
        --model-dir /home/eisen/spectralai/remote_models/Google/Gemma4-26B-A4B \\
        --model-kind gemma4 \\
        --out-dir data/gemma4_hiddens \\
        --max-tokens 200000 \\
        --quant 4bit

    python python/extract_router_io.py \\
        --model-dir /home/eisen/spectralai/remote_models/Qwen/Qwen3.6-35B-A3B \\
        --model-kind qwen36 \\
        --out-dir data/qwen36_hiddens \\
        --max-tokens 100000 \\
        --quant 4bit
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Optional

import torch


def find_router_modules(model, kind: str):
    """Return list of (layer_idx, router_module) for the chosen architecture."""
    routers = []
    if kind == "gemma4":
        # path: model.language_model.layers.N.router  (Gemma4TextRouter)
        # AutoModelForCausalLM → Gemma4ForConditionalGeneration; text branch is at .model.language_model
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            layers = model.model.language_model.layers
        else:
            # If we somehow got the text-only Gemma4ForCausalLM
            layers = model.model.layers
        for i, layer in enumerate(layers):
            if hasattr(layer, "router"):
                routers.append((i, layer.router))
    elif kind == "qwen36":
        # path: model.layers.N.mlp.gate  (Qwen3_5MoeTopKRouter)
        # AutoModelForCausalLM → Qwen3_5MoeForCausalLM (text-only)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "model") and hasattr(model.model, "language_model"):
            layers = model.model.language_model.layers
        else:
            raise RuntimeError(f"Cannot find layers in {type(model).__name__}")
        for i, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "gate"):
                routers.append((i, mlp.gate))
    else:
        raise ValueError(f"unknown kind {kind}")
    return routers


class RouterCapture:
    """Forward hook that accumulates (hidden_states, gate_logits, topk_ids) per layer.

    Field names match `RealHiddensDataset` in python/olmoe_bvh_distill.py so the
    existing training loop can consume our output directly. `gate_logits` is
    actually post-softmax probabilities (the distillation loss handles both).
    """

    def __init__(self, layer_idx: int, max_tokens: int):
        self.layer_idx = layer_idx
        self.max_tokens = max_tokens
        self.hidden_states = []  # list of (n, hidden_dim) cpu fp16 tensors
        self.gate_logits = []    # list of (n, n_experts) cpu fp16 tensors (full softmax probs)
        self.topk_ids = []       # list of (n, k) cpu int64 tensors
        self.n_collected = 0

    def hook(self, module, args, output):
        if self.n_collected >= self.max_tokens:
            return
        # Both Gemma4TextRouter and Qwen3_5MoeTopKRouter take (N, hidden) or (B,S,hidden)
        x = args[0]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        # Both return a 3-tuple; layout varies by model:
        #   Gemma4TextRouter: (router_probabilities, top_k_weights, top_k_index)
        #   Qwen3_5MoeTopKRouter: (router_logits[softmax], router_top_value, router_indices)
        # First element is always the full softmax distribution.
        if isinstance(output, tuple) and len(output) == 3:
            full_probs, _weights, indices = output
        else:
            return  # unknown signature; skip

        avail = self.max_tokens - self.n_collected
        if x.shape[0] > avail:
            x = x[:avail]
            full_probs = full_probs[:avail]
            indices = indices[:avail]

        # Downcast + move to CPU
        self.hidden_states.append(x.detach().to("cpu", dtype=torch.float16, copy=True))
        self.gate_logits.append(full_probs.detach().to("cpu", dtype=torch.float16, copy=True))
        self.topk_ids.append(indices.detach().to("cpu", dtype=torch.int64, copy=True))
        self.n_collected += x.shape[0]

    def save(self, out_path: Path):
        if not self.hidden_states:
            return False
        torch.save(
            {
                "layer_idx": self.layer_idx,
                "hidden_states": torch.cat(self.hidden_states, dim=0),  # (N, hidden) fp16
                "gate_logits":   torch.cat(self.gate_logits, dim=0),    # (N, n_experts) fp16
                "topk_ids":      torch.cat(self.topk_ids, dim=0),       # (N, k) int64
                "n_tokens": self.n_collected,
            },
            out_path,
        )
        return True


def load_dataset_iter(seq_len: int, tokenizer):
    """Stream WikiText-2 train split, yielding tokenized chunks of seq_len."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=False)
    buf = []
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        while len(buf) >= seq_len:
            chunk, buf = buf[:seq_len], buf[seq_len:]
            yield torch.tensor(chunk, dtype=torch.long).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-kind", required=True, choices=["gemma4", "qwen36"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=200_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--quant", choices=["4bit", "8bit", "bf16", "fp16"], default="4bit")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--single-layer", type=int, default=None,
                    help="If set, only collect from this layer (for fast smoke-test)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] tokenizer from {args.model_dir}")
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    print(f"[load] model ({args.quant}) from {args.model_dir}")
    t0 = time.perf_counter()

    # For Gemma 4: ConditionalGeneration wrapper bundles vision_tower + audio_tower.
    # In bf16 they bloat memory; in 4bit they don't fit on GPU because of the
    # Params4bit+CPU-offload incompat. Strategy by quant:
    #   bf16/fp16: device_map=auto with max_memory cap (some language_model on CPU)
    #   4bit: skip vision/audio from quant + keep them on CPU; language_model fully on GPU in 4bit
    if args.model_kind == "gemma4" and args.device_map == "auto":
        if args.quant == "4bit":
            # Vision + audio stay bf16 on CPU; language_model 4bit but split
            # because 26B in 4bit (13GB) + embeddings + lm_head (~2GB at bf16)
            # exceeds 16GB. Spill last several layers to CPU.
            n_layers = 30  # gemma4 26b a4b
            cpu_layers = list(range(22, n_layers))  # 22-29 → CPU
            device_map_arg = {
                "model.vision_tower": "cpu",
                "model.audio_tower": "cpu",
                "model.embed_vision": "cpu",
                "model.embed_audio": "cpu",
                "model.embed_audio_norm": "cpu",
                "model.language_model.embed_tokens": 0,
                "model.language_model.norm": "cpu",
                "lm_head": "cpu",
            }
            for i in range(n_layers):
                device_map_arg[f"model.language_model.layers.{i}"] = (
                    "cpu" if i in cpu_layers else 0
                )
            max_memory = None
            print(
                f"[load] 4bit: vision/audio→CPU; language layers 0-{cpu_layers[0]-1} on GPU; "
                f"layers {cpu_layers[0]}-{n_layers-1} + lm_head + final norm on CPU"
            )
        else:
            max_memory = {0: "12GiB", "cpu": "120GiB"}
            device_map_arg = "auto"
            print(f"[load] device_map=auto, max_memory={max_memory}")
    else:
        max_memory = None
        device_map_arg = args.device_map

    if args.quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        # Skip vision/audio modules from quant so they stay bf16 on CPU.
        skip_modules = ["vision_tower", "audio_tower", "embed_vision", "embed_audio", "lm_head"]
        if args.quant == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=skip_modules,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        else:
            bnb_cfg = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=skip_modules,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        kwargs = dict(
            quantization_config=bnb_cfg,
            device_map=device_map_arg,
            dtype=torch.bfloat16,
        )
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, **kwargs)
    else:
        dtype = torch.bfloat16 if args.quant == "bf16" else torch.float16
        kwargs = dict(dtype=dtype, device_map=device_map_arg)
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, **kwargs)
    print(f"[load] {type(model).__name__} loaded in {time.perf_counter() - t0:.1f}s")

    model.eval()

    routers = find_router_modules(model, args.model_kind)
    print(f"[hook] found {len(routers)} routers")
    if args.single_layer is not None:
        routers = [(i, m) for i, m in routers if i == args.single_layer]
        if not routers:
            raise SystemExit(f"layer {args.single_layer} has no router")

    # Per-layer cap on tokens; we'll stop early once every layer is full
    per_layer_max = args.max_tokens
    captures = {}
    handles = []
    for layer_idx, router in routers:
        cap = RouterCapture(layer_idx, per_layer_max)
        captures[layer_idx] = cap
        handles.append(router.register_forward_hook(cap.hook))

    # Iterate over tokens
    print(f"[run] forwarding chunks of {args.seq_len} tokens until each layer has >= {per_layer_max}")
    n_passed = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for chunk in load_dataset_iter(args.seq_len, tok):
            chunk = chunk.to(model.device if hasattr(model, "device") else "cuda")
            _ = model(chunk, use_cache=False)
            n_passed += chunk.shape[1]
            done = all(c.n_collected >= per_layer_max for c in captures.values())
            if n_passed % 5120 == 0 or done:
                min_collected = min(c.n_collected for c in captures.values())
                rate = n_passed / max(time.perf_counter() - t0, 1e-6)
                print(
                    f"  passed={n_passed:>7d} tokens; min_per_layer={min_collected:>7d}/{per_layer_max} "
                    f"({rate:.0f} tok/s)",
                    flush=True,
                )
            if done:
                break

    for h in handles:
        h.remove()

    # Save per-layer
    print(f"[save] writing per-layer .pt files to {args.out_dir}")
    for layer_idx, cap in captures.items():
        out_path = args.out_dir / f"router_io_layer{layer_idx:02d}.pt"
        if cap.save(out_path):
            print(f"  layer {layer_idx:>2}: {cap.n_collected} tokens → {out_path}")

    # Summary metadata
    meta = {
        "model_dir": str(args.model_dir),
        "model_kind": args.model_kind,
        "model_class": type(model).__name__,
        "n_layers_captured": len(captures),
        "tokens_per_layer": {i: c.n_collected for i, c in captures.items()},
        "seq_len": args.seq_len,
        "quant": args.quant,
        "wallclock_s": round(time.perf_counter() - t0, 2),
    }
    (args.out_dir / "extract_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[done] wrote {args.out_dir}/extract_meta.json")


if __name__ == "__main__":
    main()
