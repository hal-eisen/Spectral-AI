"""End-to-end tok/s on Gemma 4 26B A4B with experts at 4-bit on GPU + BVH routing.

This is the configuration that finally lets BVH speedup show up at the
model level: 24 of 30 layers' experts quantized to NF4 on GPU (~13 GB),
6 layers' experts on CPU as fallback. Routing happens entirely on GPU.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default="/home/eisen/spectralai/remote_models/Google/Gemma4-26B-A4B")
    ap.add_argument("--checkpoint-dir", type=Path,
                    default=Path("checkpoints/gemma4_distill_branch_200k"))
    ap.add_argument("--max-gpu-layers", type=int, default=24,
                    help="Layers whose experts get 4-bit on GPU; rest stay bf16 on CPU")
    ap.add_argument("--n-candidates", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out-md", type=Path, default=Path("results/gemma4_e2e_bvh_4bit.md"))
    ap.add_argument("--out-json", type=Path, default=Path("results/gemma4_e2e_bvh_4bit.json"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from python.gemma4_e2e_real_speedup import (
        build_surgical_device_map, snapshot_original_routers, restore_baseline_routers,
    )
    from python.quantize_experts import quantize_experts_to_4bit
    from python.gemma4_e2e_eval import install_adapters

    print(f"[load] {args.model_dir}", flush=True)
    t0 = time.perf_counter()
    dm = build_surgical_device_map(30)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map=dm,
    )
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s; VRAM "
          f"{torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)

    print(f"[quant] experts→GPU(4-bit) for first {args.max_gpu_layers} layers", flush=True)
    t1 = time.perf_counter()
    n = quantize_experts_to_4bit(model, model_dir=args.model_dir,
                                  max_gpu_layers=args.max_gpu_layers)
    print(f"  converted {n} layers in {time.perf_counter()-t1:.1f}s; VRAM "
          f"{torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)
    model.eval()

    originals = snapshot_original_routers(model)

    # ====== Baseline: original Gemma4TextRouter ======
    print(f"\n=== baseline (original gate, 4-bit experts) ===", flush=True)
    r_base = time_prefill(model, tok.vocab_size, args.batch, args.seq_len)
    print(f"  {r_base['tok_per_s']:.2f} tok/s "
          f"(median {r_base['median_s']*1000:.0f}ms ± {r_base['stdev_s']*1000:.0f}ms)",
          flush=True)

    # ====== BVH hybrid ======
    print(f"\n=== BVH hybrid n_cand={args.n_candidates}, 4-bit experts ===", flush=True)
    n_swap = install_adapters(model, args.checkpoint_dir, "hybrid",
                               args.n_candidates, model_dir=args.model_dir)
    print(f"  installed {n_swap} BVH wrappers", flush=True)
    r_bvh = time_prefill(model, tok.vocab_size, args.batch, args.seq_len)
    print(f"  {r_bvh['tok_per_s']:.2f} tok/s "
          f"(median {r_bvh['median_s']*1000:.0f}ms ± {r_bvh['stdev_s']*1000:.0f}ms)",
          flush=True)

    restore_baseline_routers(model, originals)

    speedup = r_bvh["tok_per_s"] / r_base["tok_per_s"]
    results = {
        "model": "gemma4-26b-a4b",
        "config": "experts 4-bit on GPU (24/30 layers), 6 layers' experts on CPU bf16",
        "vram_used_gb": round(torch.cuda.memory_allocated()/1024**3, 2),
        "n_candidates": args.n_candidates,
        "baseline": r_base,
        "bvh_hybrid": r_bvh,
        "speedup_bvh_vs_baseline": round(speedup, 4),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2))

    md = ["# Gemma 4 26B A4B — e2e tok/s with experts at 4-bit on GPU\n"]
    md.append(f"Config: experts NF4-quantized on GPU for first {args.max_gpu_layers} layers, "
              f"remaining {30 - args.max_gpu_layers} layers' experts on CPU bf16. "
              f"Dense parts (attention, dense MLP, routers, embeddings, lm_head) on GPU bf16. "
              f"Total VRAM used ~{results['vram_used_gb']} GB on RTX 4070 Ti Super (16 GB).\n")
    md.append(f"Prefill (B={args.batch}, S={args.seq_len}), median of "
              f"{len(r_base['all_runs_s'])} runs:\n")
    md.append("| Config | Median latency (ms) | Tok/s | Speedup |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| baseline (original gate) | {r_base['median_s']*1000:.0f} | "
              f"{r_base['tok_per_s']:.2f} | 1.00× |")
    md.append(f"| BVH hybrid n_cand={args.n_candidates} | {r_bvh['median_s']*1000:.0f} | "
              f"{r_bvh['tok_per_s']:.2f} | **{speedup:.3f}×** |")
    md.append("\n## What changed vs the experts-on-CPU baseline\n")
    md.append("- Experts at 4-bit on GPU: throughput jumped from 107.6 → "
              f"{r_base['tok_per_s']:.0f} tok/s ({r_base['tok_per_s']/107.6:.2f}× over the "
              "experts-fully-on-CPU baseline)")
    md.append("- BVH adds the routing-primitive-level speedup on top, captured in this "
              "row's tok/s improvement vs baseline")
    args.out_md.write_text("\n".join(md))
    print(f"\nwrote {args.out_md}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
