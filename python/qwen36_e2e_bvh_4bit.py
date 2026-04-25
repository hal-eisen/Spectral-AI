"""End-to-end tok/s on Qwen 3.6 35B A3B with experts at 4-bit on GPU + BVH routing.

Mirrors gemma4_e2e_bvh_4bit.py but for Qwen 3.6: Qwen3_5MoeForCausalLM,
40 MoE layers (mix of linear_attention and full_attention every 4th), 256
experts × 40 layers × 2 weight tensors = 80 expert tensors to quantize.

VRAM budget for 16 GB:
  embed (1.0 GB) + lm_head (1.0 GB) + dense per-layer (~80 MB × 40 = 3.2 GB)
  + activations (2 GB) = ~7 GB outside experts.
  Available for experts: ~9 GB.
  Per layer at 4-bit: 0.39 GB. Max ~22 layers' experts on GPU.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_qwen36_device_map(layer_types: list[str]) -> dict:
    """Dense parts on GPU, mlp.experts on CPU. Per-layer attention type
    matches the actual config so accelerate doesn't get confused.

    Note: AutoModelForCausalLM loads Qwen3_5MoeForCausalLM (text-only) which
    uses `model.layers.*` paths; the safetensors store keys under
    `model.language_model.layers.*` and transformers remaps during load. The
    device_map must use the in-memory paths.
    """
    n_layers = len(layer_types)
    dm = {
        "model.embed_tokens": 0,
        "model.norm": 0,
        "model.rotary_emb": 0,  # RoPE buffer (inv_freq) — NOT a Parameter, not in safetensors
        "lm_head": 0,
    }
    for i, lt in enumerate(layer_types):
        b = f"model.layers.{i}"
        # Per-layer attention: linear_attn (Mamba) OR self_attn (full)
        if lt == "linear_attention":
            dm[f"{b}.linear_attn"] = 0
        elif lt == "full_attention":
            dm[f"{b}.self_attn"] = 0
        else:
            raise ValueError(f"unexpected layer type: {lt}")
        # MoE block parts (Qwen3_5MoeSparseMoeBlock)
        dm[f"{b}.mlp.gate"] = 0
        dm[f"{b}.mlp.shared_expert"] = 0
        dm[f"{b}.mlp.shared_expert_gate"] = 0
        # Layer norms
        dm[f"{b}.input_layernorm"] = 0
        dm[f"{b}.post_attention_layernorm"] = 0
        # Experts → CPU initially; quantizer moves them to GPU at 4-bit
        dm[f"{b}.mlp.experts"] = "cpu"
    return dm


def time_prefill(model, vocab_size, batch, seq_len, *, warmup=2, iters=5):
    dev = next(p.device for p in model.parameters() if p.device.type == "cuda")
    ids = torch.randint(0, vocab_size, (batch, seq_len), device=dev)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(ids, use_cache=False)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(ids, use_cache=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    median = statistics.median(times)
    return {
        "median_s": round(median, 4),
        "stdev_s": round(statistics.stdev(times), 4) if len(times) > 1 else 0.0,
        "tok_per_s": round(batch * seq_len / median, 2),
        "all_runs_s": [round(t, 4) for t in times],
    }


def snapshot_qwen_routers(model):
    layers = (model.model.language_model.layers
              if hasattr(model.model, "language_model")
              else model.model.layers)
    return {i: layer.mlp.gate for i, layer in enumerate(layers)
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate")}


def restore_qwen_routers(model, originals):
    layers = (model.model.language_model.layers
              if hasattr(model.model, "language_model")
              else model.model.layers)
    for i, layer in enumerate(layers):
        if i in originals:
            layer.mlp.gate = originals[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default="/home/eisen/spectralai/remote_models/Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--checkpoint-dir", type=Path,
                    default=Path("checkpoints/qwen36_distill_branch_200k"))
    ap.add_argument("--max-gpu-layers", type=int, default=20,
                    help="Layers whose experts get 4-bit on GPU; rest stay bf16 on CPU")
    ap.add_argument("--n-candidates", type=int, default=128,
                    help="50% of 256 experts is the operating point from P7 sweep")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out-md", type=Path, default=Path("results/qwen36_e2e_bvh_4bit.md"))
    ap.add_argument("--out-json", type=Path, default=Path("results/qwen36_e2e_bvh_4bit.json"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from python.quantize_experts import quantize_experts_to_4bit
    from python.qwen36_e2e_eval import install_adapters

    print(f"[load] {args.model_dir}", flush=True)
    t0 = time.perf_counter()
    # Read layer_types from config so device_map matches per-layer attention type
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_dir)
    layer_types = (cfg.text_config.layer_types if hasattr(cfg, "text_config")
                   else cfg.layer_types)
    dm = build_qwen36_device_map(layer_types)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map=dm,
    )
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s; VRAM "
          f"{torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)

    print(f"[quant] experts→GPU(4-bit) for first {args.max_gpu_layers} layers", flush=True)
    t1 = time.perf_counter()
    n = quantize_experts_to_4bit(model, model_dir=args.model_dir,
                                  model_kind="qwen36",
                                  max_gpu_layers=args.max_gpu_layers)
    print(f"  converted {n} layers in {time.perf_counter()-t1:.1f}s; VRAM "
          f"{torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)
    model.eval()

    originals = snapshot_qwen_routers(model)

    print(f"\n=== baseline (original gate, 4-bit experts) ===", flush=True)
    r_base = time_prefill(model, tok.vocab_size, args.batch, args.seq_len)
    print(f"  {r_base['tok_per_s']:.2f} tok/s "
          f"(median {r_base['median_s']*1000:.0f}ms ± {r_base['stdev_s']*1000:.0f}ms)",
          flush=True)

    print(f"\n=== BVH hybrid n_cand={args.n_candidates}, 4-bit experts ===", flush=True)
    n_swap = install_adapters(model, args.checkpoint_dir, "hybrid", args.n_candidates)
    print(f"  installed {n_swap} BVH wrappers", flush=True)
    r_bvh = time_prefill(model, tok.vocab_size, args.batch, args.seq_len)
    print(f"  {r_bvh['tok_per_s']:.2f} tok/s "
          f"(median {r_bvh['median_s']*1000:.0f}ms ± {r_bvh['stdev_s']*1000:.0f}ms)",
          flush=True)

    restore_qwen_routers(model, originals)

    speedup = r_bvh["tok_per_s"] / r_base["tok_per_s"]
    results = {
        "model": "qwen3.6-35b-a3b",
        "config": (f"experts NF4 on GPU ({args.max_gpu_layers}/40 layers), "
                   f"remaining layers' experts on CPU bf16"),
        "vram_used_gb": round(torch.cuda.memory_allocated()/1024**3, 2),
        "n_candidates": args.n_candidates,
        "baseline": r_base,
        "bvh_hybrid": r_bvh,
        "speedup_bvh_vs_baseline": round(speedup, 4),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2))

    md = ["# Qwen 3.6 35B A3B — e2e tok/s with experts at 4-bit on GPU\n"]
    md.append(f"Config: experts NF4-quantized on GPU for first {args.max_gpu_layers} of 40 "
              f"layers, remaining {40 - args.max_gpu_layers} layers' experts on CPU bf16. "
              f"Dense parts (linear/self attention, routers, shared experts, embed, "
              f"lm_head) on GPU bf16. Total VRAM ~{results['vram_used_gb']} GB on RTX "
              f"4070 Ti Super (16 GB).\n")
    md.append(f"Prefill (B={args.batch}, S={args.seq_len}), median of "
              f"{len(r_base['all_runs_s'])} runs:\n")
    md.append("| Config | Median latency (ms) | Tok/s | Speedup |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| baseline (original gate) | {r_base['median_s']*1000:.0f} | "
              f"{r_base['tok_per_s']:.2f} | 1.00× |")
    md.append(f"| BVH hybrid n_cand={args.n_candidates} | {r_bvh['median_s']*1000:.0f} | "
              f"{r_bvh['tok_per_s']:.2f} | **{speedup:.3f}×** |")
    args.out_md.write_text("\n".join(md))
    print(f"\nwrote {args.out_md}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
