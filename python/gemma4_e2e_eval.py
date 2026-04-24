"""End-to-end PPL + tok/s evaluation for Gemma 4 26B A4B with BVH routing.

Replaces each layer's `Gemma4TextRouter` with a BVHRouterAdapter that
preserves the original router's calibration (RMSNorm + scale + per_expert_scale)
but swaps the `.proj` linear (hidden -> n_experts) step for the trained
BranchSpecificBVHRouter.

Two modes:
  --mode pure     BVH selects top-k experts directly
  --mode hybrid   BVH selects N candidates; original .proj scores those for top-k

Usage (after extraction + training):
  python python/gemma4_e2e_eval.py \
    --model-dir /home/eisen/spectralai/remote_models/Google/Gemma4-26B-A4B \
    --checkpoint-dir checkpoints/gemma4_distill_branch \
    --mode hybrid --n-candidates 32 \
    --eval-mode ppl --max-chunks 50

  python python/gemma4_e2e_eval.py \
    --model-dir /home/eisen/spectralai/remote_models/Google/Gemma4-26B-A4B \
    --checkpoint-dir checkpoints/gemma4_distill_branch \
    --eval-mode throughput --batch-size 1 --seq-len 512
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_bvh_router(ckpt_path: Path, device: torch.device) -> nn.Module:
    """Load a trained BranchSpecificBVHRouter from a checkpoint produced by
    olmoe_bvh_distill.py."""
    from python.bvh_router import BranchSpecificBVHRouter, RouterConfig

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = RouterConfig(
        embed_dim=cfg_dict["input_dim"],
        n_level1=cfg_dict["n_level1"],
        n_level2=cfg_dict["n_level2"],
        n_level3=cfg_dict["n_level3"],
        spectral_dim=cfg_dict.get("spectral_dim", 64),
    )
    router = BranchSpecificBVHRouter(cfg)
    router.load_state_dict(ckpt["router_state_dict"])
    router.eval()
    return router.to(device)


class BVHRouterAdapter(nn.Module):
    """Drop-in replacement for Gemma4TextRouter.

    Preserves: RMSNorm + scale + per_expert_scale (the calibration of the
    pretrained router). Replaces only the .proj linear (hidden -> n_experts)
    with the trained BVH router.

    Forward returns the same 3-tuple shape contract:
        (router_probabilities, top_k_weights, top_k_index)
    """

    def __init__(
        self,
        original_router: nn.Module,
        bvh_router: nn.Module,
        mode: str = "hybrid",
        n_candidates: int = 32,
        top_k_experts: Optional[int] = None,
    ):
        super().__init__()
        # Inherit the calibration components verbatim
        self.norm = original_router.norm
        self.scale = original_router.scale
        self.per_expert_scale = original_router.per_expert_scale
        self.scalar_root_size = original_router.scalar_root_size
        self.eps = original_router.eps
        self.config = original_router.config
        # Original linear (only used in hybrid mode for accurate scoring)
        self.original_proj = original_router.proj
        # BVH router (replaces proj for routing decisions)
        self.bvh = bvh_router

        self.mode = mode
        self.n_candidates = n_candidates
        self.top_k = top_k_experts or original_router.config.top_k_experts
        self.num_experts = original_router.config.num_experts

    def forward(self, hidden_states: torch.Tensor):
        # Match original router's preprocessing exactly
        x = self.norm(hidden_states)
        x = x * self.scale * self.scalar_root_size

        # Step 1: BVH router scores all experts (soft probs)
        with torch.no_grad():
            bvh_probs, _ = self.bvh(x.float())

        if self.mode == "pure":
            # Use BVH probs directly
            full_probs = F.softmax(bvh_probs, dim=-1).to(x.dtype)
            top_k_weights, top_k_index = torch.topk(full_probs, self.top_k, dim=-1)
        elif self.mode == "hybrid":
            # BVH narrows to N candidates; original .proj scores those exactly
            _, candidate_ids = torch.topk(bvh_probs, self.n_candidates, dim=-1)
            full_logits = self.original_proj(x)
            full_probs = F.softmax(full_logits, dim=-1)
            cand_probs = full_probs.gather(1, candidate_ids)
            top_k_vals, top_k_local = torch.topk(cand_probs, self.top_k, dim=-1)
            top_k_index = candidate_ids.gather(1, top_k_local)
            top_k_weights = top_k_vals
        else:
            raise ValueError(f"unknown mode {self.mode}")

        # Renormalize + apply per_expert_scale, matching the original
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return full_probs, top_k_weights, top_k_index


def install_adapters(model, checkpoint_dir: Path, mode: str, n_candidates: int):
    """Walk Gemma 4's MoE layers and replace each .router with BVHRouterAdapter.
    Returns the count of adapted layers."""
    layers = model.model.language_model.layers
    n_adapted = 0
    device = next(model.parameters()).device
    for i, layer in enumerate(layers):
        if not hasattr(layer, "router"):
            continue
        ckpt = checkpoint_dir / f"bvh_router_L{i}_best.pt"
        if not ckpt.exists():
            print(f"  [skip] layer {i}: no checkpoint at {ckpt}", file=sys.stderr)
            continue
        bvh = _load_bvh_router(ckpt, device)
        # Need bvh on the SAME device as the original router (which may be CPU
        # for offloaded layers). Use the layer's router device:
        router_dev = next(layer.router.parameters()).device
        bvh = bvh.to(router_dev)
        adapter = BVHRouterAdapter(
            original_router=layer.router,
            bvh_router=bvh,
            mode=mode,
            n_candidates=n_candidates,
        )
        layer.router = adapter
        n_adapted += 1
    return n_adapted


def load_model(model_dir: str, quant: str = "bf16"):
    """Load Gemma 4 26B A4B with the same settings extract_router_io.py uses."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    if quant == "bf16":
        max_memory = {0: "12GiB", "cpu": "120GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map="auto", max_memory=max_memory
        )
    else:
        raise NotImplementedError(f"quant={quant} not supported in eval yet")
    model.eval()
    return model, tok


@torch.no_grad()
def eval_ppl(model, tok, max_chunks: int = 50, ctx: int = 512):
    """Compute WikiText-2 perplexity on `max_chunks` chunks of `ctx` tokens.

    Mirrors llama.cpp's perplexity tool semantics: stride = ctx, no overlap,
    cross-entropy on shifted-by-1 targets.
    """
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
    device = next(model.parameters()).device

    for i in range(n_chunks):
        chunk = ids[i * ctx : (i + 1) * ctx].unsqueeze(0).to(device)
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
        if (i + 1) % 5 == 0 or i + 1 == n_chunks:
            mean_nll = total_loss / total_tokens
            ppl_so_far = math.exp(mean_nll)
            print(f"  chunk {i+1}/{n_chunks}  PPL={ppl_so_far:.4f}", flush=True)

    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def eval_throughput(model, tok, batch_size: int, seq_len: int, n_iters: int = 5,
                    warmup: int = 2):
    """Prefill tok/s using the same harness as throughput_bench."""
    from python.throughput_bench import measure_prefill_tok_per_sec

    def fwd(ids):
        return model(ids, use_cache=False).logits

    return measure_prefill_tok_per_sec(
        fwd,
        label=f"gemma4_prefill_b{batch_size}_s{seq_len}",
        vocab_size=tok.vocab_size,
        batch_size=batch_size,
        seq_len=seq_len,
        warmup_iters=warmup,
        n_iters=n_iters,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="If unset, runs baseline (no BVH swap)")
    ap.add_argument("--mode", choices=["pure", "hybrid"], default="hybrid")
    ap.add_argument("--n-candidates", type=int, default=32)
    ap.add_argument("--quant", choices=["bf16"], default="bf16")
    ap.add_argument("--eval-mode", choices=["ppl", "throughput"], required=True)
    ap.add_argument("--max-chunks", type=int, default=50)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    print(f"[load] {args.model_dir} ({args.quant})")
    t0 = time.perf_counter()
    model, tok = load_model(args.model_dir, args.quant)
    print(f"[load] done in {time.perf_counter() - t0:.1f}s ({type(model).__name__})")

    if args.checkpoint_dir is not None:
        print(f"[swap] installing BVH adapters from {args.checkpoint_dir} (mode={args.mode})")
        n = install_adapters(model, args.checkpoint_dir, args.mode, args.n_candidates)
        print(f"[swap] adapted {n} layers")
    else:
        print("[swap] skipped — running BASELINE (no BVH)")

    result = {
        "model_dir": args.model_dir,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "mode": args.mode if args.checkpoint_dir else "baseline",
        "n_candidates": args.n_candidates if args.mode == "hybrid" else None,
        "eval_mode": args.eval_mode,
    }

    if args.eval_mode == "ppl":
        print(f"[ppl] {args.max_chunks} chunks of {args.ctx} tokens")
        ppl = eval_ppl(model, tok, max_chunks=args.max_chunks, ctx=args.ctx)
        result["ppl"] = ppl
        result["max_chunks"] = args.max_chunks
        result["ctx"] = args.ctx
        print(f"\nFINAL PPL: {ppl:.4f}")
    else:
        r = eval_throughput(model, tok, args.batch_size, args.seq_len)
        result.update(r.to_row())
        print(f"\nFINAL tok/s: median={r.median_tok_per_sec:.2f}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        # Append-mode: load existing list, append, write back
        existing = []
        if args.out_json.exists():
            try:
                existing = json.loads(args.out_json.read_text())
            except Exception:
                existing = []
        existing.append(result)
        args.out_json.write_text(json.dumps(existing, indent=2, default=str))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
