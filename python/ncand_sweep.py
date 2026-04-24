"""n_candidates sweep for a trained BVH router — reuses one model load.

Loads the model once, installs the BVH-hybrid adapters at the maximum
n_candidates we'll sweep (so all BVH ranking info is available each forward),
then mutates each wrapper's `.n_candidates` in-place between eval runs and
re-measures PPL. Per extra sweep point the cost is just the PPL pass
(~13 min for Gemma 4, ~13 min for Qwen 3.6 each), not a full model reload
(~3 min each).

Usage:
  python python/ncand_sweep.py --model gemma4 --values 32 48 64 96 128
  python python/ncand_sweep.py --model qwen36 --values 32 64 96 128 192
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


def _eval_ppl_silent(model, tok, max_chunks: int, ctx: int, label: str) -> float:
    """Same PPL loop as the e2e eval scripts, but prints a label with each chunk."""
    from datasets import load_dataset

    text = "\n\n".join(
        row["text"] for row in load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
        )
    )
    ids = tok(text, return_tensors="pt").input_ids[0]
    n_chunks = min(max_chunks, ids.shape[0] // ctx)
    total_loss = 0.0
    total_tokens = 0
    dev = next(model.parameters()).device

    with torch.no_grad():
        for i in range(n_chunks):
            chunk = ids[i * ctx : (i + 1) * ctx].unsqueeze(0).to(dev)
            out = model(chunk, use_cache=False)
            logits = out.logits[..., :-1, :].contiguous()
            targets = chunk[..., 1:].contiguous()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
            if (i + 1) % 10 == 0 or i + 1 == n_chunks:
                print(f"    [{label}] chunk {i+1}/{n_chunks} PPL={math.exp(total_loss/total_tokens):.4f}", flush=True)
    return math.exp(total_loss / total_tokens)


def set_all_ncandidates(model, n: int, model_kind: str):
    """Mutate every installed BVH wrapper's n_candidates in-place."""
    if model_kind == "gemma4":
        layers = model.model.language_model.layers
        for layer in layers:
            router = layer.router
            if hasattr(router, "n_candidates"):  # it's our BVH wrapper
                router.n_candidates = n
    else:  # qwen36
        layers = model.model.layers
        for layer in layers:
            gate = layer.mlp.gate
            if hasattr(gate, "n_candidates"):
                gate.n_candidates = n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gemma4", "qwen36"], required=True)
    ap.add_argument("--values", type=int, nargs="+", required=True,
                    help="n_candidates values to sweep (ascending)")
    ap.add_argument("--max-chunks", type=int, default=50)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--ckpt-dir", type=str, default=None,
                    help="Override default checkpoints/<model>_distill_branch")
    args = ap.parse_args()

    if args.model == "gemma4":
        from python.gemma4_e2e_eval import load_model, install_adapters
        model_dir = "/home/eisen/spectralai/remote_models/Google/Gemma4-26B-A4B"
        default_ckpt = "checkpoints/gemma4_distill_branch"
    else:
        from python.qwen36_e2e_eval import load_model, install_adapters
        model_dir = "/home/eisen/spectralai/remote_models/Qwen/Qwen3.6-35B-A3B"
        default_ckpt = "checkpoints/qwen36_distill_branch"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(default_ckpt)

    out_md = args.out_md or Path(f"results/{args.model}_ncand_sweep.md")
    out_json = args.out_json or Path(f"results/{args.model}_ncand_sweep.json")

    print(f"[sweep] {args.model}: n_candidates = {args.values}")
    print(f"[load] {model_dir}")
    t0 = time.perf_counter()
    model, tok = load_model(model_dir)
    print(f"[load] done in {time.perf_counter()-t0:.1f}s ({type(model).__name__})")

    # Install once with the largest n_candidates (doesn't matter which — we
    # reset per iteration). Use max so if any install-time path depended on it,
    # the other values are safe.
    install_n = max(args.values)
    print(f"[swap] installing adapters (initial n_candidates={install_n})")
    if args.model == "gemma4":
        n_adapted = install_adapters(model, ckpt_dir, "hybrid", install_n,
                                      model_dir=model_dir)
    else:
        n_adapted = install_adapters(model, ckpt_dir, "hybrid", install_n)
    print(f"[swap] {n_adapted} layers adapted")

    results = []
    for n_cand in sorted(args.values):
        set_all_ncandidates(model, n_cand, args.model)
        print(f"\n=== n_candidates = {n_cand} ===")
        t0 = time.perf_counter()
        ppl = _eval_ppl_silent(model, tok, args.max_chunks, args.ctx,
                               label=f"ncand={n_cand}")
        dt = time.perf_counter() - t0
        print(f"  FINAL PPL = {ppl:.4f}  ({dt:.0f}s)")
        results.append({
            "n_candidates": n_cand,
            "ppl": round(ppl, 4),
            "eval_seconds": round(dt, 1),
        })

    # Also a "pure BVH" row (n_candidates irrelevant — just sets mode)
    # for completeness: iterate all layers and set mode to "pure"
    print("\n=== mode=pure (BVH only, no original scoring) ===")
    for layer in (model.model.language_model.layers if args.model == "gemma4"
                  else model.model.layers):
        holder = layer.router if args.model == "gemma4" else layer.mlp.gate
        if hasattr(holder, "mode"):
            holder.mode = "pure"
    t0 = time.perf_counter()
    ppl_pure = _eval_ppl_silent(model, tok, args.max_chunks, args.ctx, label="pure")
    results.append({"n_candidates": None, "mode": "pure", "ppl": round(ppl_pure, 4),
                    "eval_seconds": round(time.perf_counter() - t0, 1)})
    print(f"  FINAL PPL (pure) = {ppl_pure:.4f}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"model": args.model, "results": results}, indent=2))

    # Build markdown
    n_experts = 128 if args.model == "gemma4" else 256
    # Load baseline PPL from prior e2e JSON if available
    baseline_ppl = None
    prior_json = Path(f"results/{args.model}_e2e.json")
    if prior_json.exists():
        try:
            prior = json.loads(prior_json.read_text())
            for e in prior:
                if e.get("checkpoint_dir") is None and e.get("eval_mode") == "ppl":
                    baseline_ppl = e.get("ppl")
                    break
        except Exception:
            pass

    md = [f"# n_candidates sweep — {args.model}\n"]
    md.append(f"- Model: {args.model} ({n_experts} experts, hidden "
              f"{2816 if args.model == 'gemma4' else 2048})")
    md.append(f"- Checkpoints: {ckpt_dir}")
    md.append(f"- Eval: WikiText-2 test, ctx={args.ctx}, max_chunks={args.max_chunks}")
    if baseline_ppl is not None:
        md.append(f"- Baseline PPL (no BVH): **{baseline_ppl:.4f}**")
    md.append("")
    md.append("| n_candidates | % of experts | PPL | Δ vs baseline |")
    md.append("|---:|---:|---:|---:|")
    for r in results:
        n = r["n_candidates"]
        if n is None:
            tag = "pure BVH"
            pct = "—"
        else:
            tag = str(n)
            pct = f"{100*n/n_experts:.1f}%"
        d = ""
        if baseline_ppl is not None:
            d = f"{100*(r['ppl']/baseline_ppl - 1):+.2f}%"
        md.append(f"| {tag} | {pct} | {r['ppl']:.4f} | {d} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n")
    print(f"\nwrote {out_md}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
