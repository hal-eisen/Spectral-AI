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
    """Load a trained EnhancedBVHRouter from a checkpoint produced by
    olmoe_bvh_distill.py (which currently only supports EnhancedBVHRouter
    end-to-end; BranchSpecificBVHRouter wiring in the training loop is TBD)."""
    from python.olmoe_bvh_distill import EnhancedBVHRouter

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    sd = ckpt["router_state_dict"]
    spectral_mode = cfg_dict.get("spectral_mode", ckpt.get("spectral_mode", False))
    spectral_dim = cfg_dict.get("spectral_dim", 64)
    enc_hidden = None
    if spectral_mode and "spectral_encoder.0.weight" in sd:
        enc_hidden = sd["spectral_encoder.0.weight"].shape[0]
        spectral_dim = sd["spectral_encoder.2.weight"].shape[0]

    router = EnhancedBVHRouter(
        input_dim=cfg_dict["input_dim"],
        n_level1=cfg_dict["n_level1"],
        n_level2=cfg_dict["n_level2"],
        n_level3=cfg_dict["n_level3"],
        feature_dim=cfg_dict.get("feature_dim", 128),
        spectral_mode=spectral_mode,
        spectral_dim=spectral_dim,
        encoder_hidden=enc_hidden,
    )
    router.load_state_dict(sd)
    router.eval()
    return router.to(device)


class BVHRoutedRouterWrapper(nn.Module):
    """Wraps the original Gemma4TextRouter.

    Forward calls the original router (so accelerate's offload hooks fire
    correctly to bring its params on-device), gets the full softmax probs,
    then uses BVH to select top-k experts. In hybrid mode the original probs
    score BVH candidates exactly; in pure mode BVH probs replace originals.
    """

    def __init__(self, original_router: nn.Module, bvh: nn.Module,
                 mode: str, n_candidates: int):
        super().__init__()
        self.original = original_router
        self.bvh = bvh
        self.mode = mode
        self.n_candidates = n_candidates
        self.top_k = original_router.config.top_k_experts
        self.num_experts = original_router.config.num_experts

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor):
        # Always call the original router → triggers accelerate's pre-forward hook
        # and gives us the full softmax probabilities + per_expert_scale-applied
        # weights. We override only the top-k selection.
        full_probs, orig_top_k_weights, orig_top_k_index = self.original(hidden_states)

        # Run BVH on the same hidden states
        bvh_dev = next(self.bvh.parameters()).device
        h = hidden_states.detach().to(device=bvh_dev, dtype=torch.float32)
        if h.dim() == 3:
            h = h.reshape(-1, h.shape[-1])
        bvh_probs, _ = self.bvh(h)

        if self.mode == "pure":
            # BVH selects top-k directly from its own probs
            top_k_weights, top_k_index = torch.topk(
                bvh_probs.to(full_probs.device), self.top_k, dim=-1
            )
            top_k_weights = top_k_weights.to(full_probs.dtype)
        elif self.mode == "hybrid":
            # BVH narrows to N candidates; original full_probs scores them exactly
            _, candidate_ids = torch.topk(bvh_probs, self.n_candidates, dim=-1)
            candidate_ids = candidate_ids.to(full_probs.device)
            cand_probs = full_probs.gather(1, candidate_ids)
            top_k_vals, top_k_local = torch.topk(cand_probs, self.top_k, dim=-1)
            top_k_index = candidate_ids.gather(1, top_k_local)
            top_k_weights = top_k_vals
        else:
            raise ValueError(f"unknown mode {self.mode}")

        # Renormalize + apply per_expert_scale (matches original Gemma4TextRouter)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        per_exp_scale = self.original.per_expert_scale[top_k_index]
        top_k_weights = top_k_weights * per_exp_scale
        return full_probs, top_k_weights, top_k_index


def install_adapters(model, checkpoint_dir: Path, mode: str, n_candidates: int):
    """Replace each MoE layer's .router with BVHRoutedRouterWrapper. The
    wrapper holds the original router as a child so accelerate continues to
    manage its offload hooks transparently."""
    layers = model.model.language_model.layers
    n_adapted = 0
    for i, layer in enumerate(layers):
        if not hasattr(layer, "router"):
            continue
        ckpt = checkpoint_dir / f"bvh_router_L{i}_best.pt"
        if not ckpt.exists():
            print(f"  [skip] layer {i}: no checkpoint at {ckpt}", file=sys.stderr)
            continue
        target_dev = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        bvh = _load_bvh_router(ckpt, target_dev)
        wrapper = BVHRoutedRouterWrapper(layer.router, bvh, mode, n_candidates)
        layer.router = wrapper
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
