#!/usr/bin/env python3
"""Aggregate per-layer router training accuracy into a markdown table.

Reads bvh_router_L{N}_best.pt files in a checkpoint directory and emits:
  - per-layer top-1 / top-8 accuracy
  - mean / min / max / stddev
  - layer-wise plot-friendly CSV (optional)

Usage:
  python scripts/aggregate_router_accuracy.py checkpoints/gemma4_distill_branch
  python scripts/aggregate_router_accuracy.py checkpoints/qwen36_distill_branch \\
      --out results/qwen36_router_acc.md
"""
import argparse
import statistics
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.ckpt_dir.glob("bvh_router_L*_best.pt"),
                   key=lambda p: int(p.stem.split("L")[1].split("_")[0]))
    if not files:
        raise SystemExit(f"no bvh_router_L*_best.pt files in {args.ckpt_dir}")

    rows = []
    for f in files:
        L = int(f.stem.split("L")[1].split("_")[0])
        ckpt = torch.load(f, weights_only=False)
        rows.append({
            "layer": L,
            "top1": ckpt["top1_accuracy"],
            "top8": ckpt["topk_accuracy"],
            "epoch": ckpt.get("epoch", "?"),
            "n_experts": ckpt["config"]["n_experts"],
            "input_dim": ckpt["config"]["input_dim"],
        })

    n_experts = rows[0]["n_experts"]
    input_dim = rows[0]["input_dim"]
    top1s = [r["top1"] for r in rows]
    top8s = [r["top8"] for r in rows]
    rand_top1 = 1.0 / n_experts
    rand_top8 = 8.0 / n_experts  # expected overlap of random top-8 with any other top-8

    # Markdown
    md = []
    md.append(f"# BVH Router accuracy — {args.ckpt_dir.name}\n")
    md.append(f"- Layers trained: {len(rows)}")
    md.append(f"- Experts: {n_experts}, hidden_dim: {input_dim}")
    md.append(f"- Random baseline: top-1 {rand_top1*100:.2f}%, top-8 overlap {rand_top8*100:.2f}%")
    md.append("")
    md.append("| Layer | top-1  | top-8  | epoch |")
    md.append("|------:|-------:|-------:|------:|")
    for r in rows:
        md.append(f"| {r['layer']:>5} | {r['top1']*100:5.1f}% | {r['top8']*100:5.1f}% | {r['epoch']} |")
    md.append(f"| **mean** | **{statistics.fmean(top1s)*100:.1f}%** "
              f"| **{statistics.fmean(top8s)*100:.1f}%** |  |")
    md.append(f"| min   | {min(top1s)*100:.1f}% | {min(top8s)*100:.1f}% |  |")
    md.append(f"| max   | {max(top1s)*100:.1f}% | {max(top8s)*100:.1f}% |  |")
    if len(rows) > 1:
        md.append(f"| stdev | {statistics.stdev(top1s)*100:.1f}% | {statistics.stdev(top8s)*100:.1f}% |  |")

    out = "\n".join(md) + "\n"
    print(out)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out)
        print(f"\nwrote {args.out}", flush=True)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w") as fh:
            fh.write("layer,top1,top8,epoch\n")
            for r in rows:
                fh.write(f"{r['layer']},{r['top1']:.4f},{r['top8']:.4f},{r['epoch']}\n")
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
