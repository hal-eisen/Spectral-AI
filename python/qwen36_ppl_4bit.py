"""PPL fidelity check for Qwen 3.6 35B A3B with 4-bit experts on GPU.

Mirror of gemma4_ppl_4bit.py for Qwen 3.6.
Measures three PPLs on the same WikiText-2 chunks:
  full-bf16 baseline                              (cached: 7.873)
  4-bit experts + original gate                   (this run)
  4-bit experts + BVH hybrid (n_cand=128, 200k)   (this run)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@torch.no_grad()
def eval_ppl(model, tok, max_chunks=50, ctx=512, label="ppl"):
    from datasets import load_dataset
    text = "\n\n".join(
        row["text"] for row in load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
        )
    )
    ids = tok(text, return_tensors="pt").input_ids[0]
    n_chunks = min(max_chunks, ids.shape[0] // ctx)
    total_loss, total_tokens = 0.0, 0
    dev = next(p.device for p in model.parameters() if p.device.type == "cuda")
    for i in range(n_chunks):
        chunk = ids[i * ctx : (i + 1) * ctx].unsqueeze(0).to(dev)
        out = model(chunk, use_cache=False)
        logits = out.logits[..., :-1, :].contiguous()
        targets = chunk[..., 1:].contiguous()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1), reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += targets.numel()
        if (i + 1) % 10 == 0 or i + 1 == n_chunks:
            ppl = math.exp(total_loss / total_tokens)
            print(f"  [{label}] chunk {i+1}/{n_chunks} PPL={ppl:.4f}", flush=True)
    return math.exp(total_loss / total_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default="/home/eisen/spectralai/remote_models/Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--checkpoint-dir", type=Path,
                    default=Path("checkpoints/qwen36_distill_branch_200k"))
    ap.add_argument("--max-gpu-layers", type=int, default=20)
    ap.add_argument("--n-candidates", type=int, default=128)
    ap.add_argument("--max-chunks", type=int, default=50)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--out-json", type=Path, default=Path("results/qwen36_ppl_4bit.json"))
    ap.add_argument("--out-md", type=Path, default=Path("results/qwen36_ppl_4bit.md"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from python.qwen36_e2e_bvh_4bit import (
        build_qwen36_device_map, snapshot_qwen_routers, restore_qwen_routers,
    )
    from python.quantize_experts import quantize_experts_to_4bit
    from python.qwen36_e2e_eval import install_adapters

    print(f"[load] {args.model_dir}", flush=True)
    t0 = time.perf_counter()
    cfg = AutoConfig.from_pretrained(args.model_dir)
    layer_types = (cfg.text_config.layer_types if hasattr(cfg, "text_config")
                   else cfg.layer_types)
    dm = build_qwen36_device_map(layer_types)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map=dm,
    )
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"[quant] experts→4-bit (first {args.max_gpu_layers} layers)", flush=True)
    t1 = time.perf_counter()
    quantize_experts_to_4bit(model, model_dir=args.model_dir,
                              model_kind="qwen36",
                              max_gpu_layers=args.max_gpu_layers)
    print(f"  done in {time.perf_counter()-t1:.1f}s; VRAM "
          f"{torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)
    model.eval()

    originals = snapshot_qwen_routers(model)

    # === PPL 1: 4-bit experts + original gate ===
    print(f"\n=== PPL: 4-bit experts + original gate ===", flush=True)
    ppl_4bit_gate = eval_ppl(model, tok, args.max_chunks, args.ctx, label="4bit_gate")
    print(f"  FINAL PPL (4-bit experts, orig gate) = {ppl_4bit_gate:.4f}", flush=True)

    # === PPL 2: 4-bit experts + BVH hybrid ===
    print(f"\n=== PPL: 4-bit experts + BVH hybrid n={args.n_candidates} ===", flush=True)
    install_adapters(model, args.checkpoint_dir, "hybrid", args.n_candidates)
    ppl_4bit_bvh = eval_ppl(model, tok, args.max_chunks, args.ctx, label="4bit_bvh")
    print(f"  FINAL PPL (4-bit experts, BVH n={args.n_candidates}) = {ppl_4bit_bvh:.4f}", flush=True)

    restore_qwen_routers(model, originals)

    # Reference PPLs from earlier runs (results/qwen36_e2e.json + ncand_sweep_200k)
    full_bf16_baseline = 7.8729
    full_bf16_bvh_n128_200k = 7.8744  # from ncand_sweep_200k

    results = {
        "model": "qwen3.6-35b-a3b",
        "config_4bit": f"experts NF4 on GPU for {args.max_gpu_layers}/40 layers",
        "vram_gb": round(torch.cuda.memory_allocated()/1024**3, 2),
        "n_candidates": args.n_candidates,
        "ctx": args.ctx, "max_chunks": args.max_chunks,
        "ppl_full_bf16_baseline": full_bf16_baseline,
        "ppl_full_bf16_bvh_n128_200k": full_bf16_bvh_n128_200k,
        "ppl_4bit_experts_baseline": round(ppl_4bit_gate, 4),
        "ppl_4bit_experts_bvh_n128_200k": round(ppl_4bit_bvh, 4),
        "delta_4bit_vs_bf16_baseline_pct": round(
            (ppl_4bit_gate / full_bf16_baseline - 1) * 100, 2),
        "delta_bvh_vs_4bit_baseline_pct": round(
            (ppl_4bit_bvh / ppl_4bit_gate - 1) * 100, 2),
        "delta_4bit_bvh_vs_bf16_baseline_pct": round(
            (ppl_4bit_bvh / full_bf16_baseline - 1) * 100, 2),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2))

    md = ["# Qwen 3.6 35B A3B — PPL fidelity with 4-bit experts on GPU\n"]
    md.append("Same WikiText-2 chunks (50, ctx=512) as the full-bf16 baseline. "
              "Tests whether the 4-bit-experts deployment configuration retains "
              "fidelity to the original safetensor model.\n")
    md.append("| Config | PPL | Δ vs full-bf16 baseline |")
    md.append("|---|---:|---:|")
    md.append(f"| Full bf16, original gate | {full_bf16_baseline:.4f} | baseline |")
    md.append(f"| Full bf16, BVH hybrid n=128 (200k) | {full_bf16_bvh_n128_200k:.4f} | "
              f"{(full_bf16_bvh_n128_200k/full_bf16_baseline-1)*100:+.2f}% |")
    md.append(f"| **4-bit experts, original gate** | **{ppl_4bit_gate:.4f}** | "
              f"**{(ppl_4bit_gate/full_bf16_baseline-1)*100:+.2f}%** |")
    md.append(f"| **4-bit experts, BVH hybrid n=128 (200k)** | **{ppl_4bit_bvh:.4f}** | "
              f"**{(ppl_4bit_bvh/full_bf16_baseline-1)*100:+.2f}%** |")
    md.append("")
    md.append("## Summary\n")
    md.append(f"- 4-bit experts costs **{(ppl_4bit_gate/full_bf16_baseline-1)*100:.2f}%** "
              "PPL vs the full-bf16 baseline.")
    md.append(f"- BVH on top of 4-bit experts: "
              f"**{(ppl_4bit_bvh/ppl_4bit_gate-1)*100:+.2f}%** additional Δ.")
    md.append(f"- Combined: 4-bit experts + BVH n=128 is "
              f"**{(ppl_4bit_bvh/full_bf16_baseline-1)*100:+.2f}%** vs the original "
              "safetensor baseline.")
    args.out_md.write_text("\n".join(md))
    print(f"\nwrote {args.out_md}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
