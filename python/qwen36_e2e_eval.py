"""End-to-end PPL + tok/s evaluation for Qwen 3.6 35B A3B with BVH routing.

Replaces each layer's `Qwen3_5MoeTopKRouter` (the MoE gate) with a BVH adapter
that preserves the router's contract: (router_logits[softmax], routing_weights,
selected_experts). The shared-expert path of Qwen3_5MoeSparseMoeBlock is
untouched.

Differs from gemma4_e2e_eval.py because Qwen 3.6's router is a much simpler
module (just a Linear-equivalent), and the parent MoE block has a separate
shared expert that we leave intact.
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


class BVHQwenRouterAdapter(nn.Module):
    """Drop-in replacement for Qwen3_5MoeTopKRouter.

    Preserves the contract:
        forward(hidden_states) -> (router_logits[softmax], top_k_weights, top_k_indices)

    In hybrid mode the original .weight stays around to score BVH candidates
    exactly. In pure mode the BVH probs become the router output directly.
    """

    def __init__(
        self,
        original_gate: nn.Module,
        bvh_router: nn.Module,
        mode: str = "hybrid",
        n_candidates: int = 32,
    ):
        super().__init__()
        self.top_k = original_gate.top_k
        self.num_experts = original_gate.num_experts
        self.hidden_dim = original_gate.hidden_dim
        # Keep the original linear weight for hybrid scoring
        self.original_weight = original_gate.weight  # (num_experts, hidden_dim)
        self.bvh = bvh_router
        self.mode = mode
        self.n_candidates = n_candidates

    def forward(self, hidden_states):
        x = hidden_states.reshape(-1, self.hidden_dim)
        with torch.no_grad():
            bvh_probs, _ = self.bvh(x.float())

        if self.mode == "pure":
            # bvh_probs is already a normalized distribution (sums to 1)
            full_probs = bvh_probs.to(x.dtype)
            top_vals, top_idx = torch.topk(full_probs, self.top_k, dim=-1)
        elif self.mode == "hybrid":
            _, candidate_ids = torch.topk(bvh_probs, self.n_candidates, dim=-1)
            full_logits = F.linear(x, self.original_weight)
            full_probs = F.softmax(full_logits, dtype=torch.float, dim=-1)
            cand_probs = full_probs.gather(1, candidate_ids)
            top_vals_cand, top_local = torch.topk(cand_probs, self.top_k, dim=-1)
            top_idx = candidate_ids.gather(1, top_local)
            top_vals = top_vals_cand
        else:
            raise ValueError(f"unknown mode {self.mode}")

        # Renormalize so per-token weights sum to 1 (matches original Qwen3_5MoeTopKRouter)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        top_vals = top_vals.to(full_probs.dtype)
        return full_probs, top_vals, top_idx


def install_adapters(model, checkpoint_dir: Path, mode: str, n_candidates: int):
    layers = model.model.layers
    n_adapted = 0
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "gate"):
            continue
        ckpt = checkpoint_dir / f"bvh_router_L{i}_best.pt"
        if not ckpt.exists():
            print(f"  [skip] layer {i}: no checkpoint at {ckpt}", file=sys.stderr)
            continue
        gate_dev = next(mlp.gate.parameters()).device
        bvh = _load_bvh_router(ckpt, gate_dev)
        adapter = BVHQwenRouterAdapter(
            original_gate=mlp.gate,
            bvh_router=bvh,
            mode=mode,
            n_candidates=n_candidates,
        )
        mlp.gate = adapter
        n_adapted += 1
    return n_adapted


def load_model(model_dir: str, quant: str = "bf16"):
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
            ppl_so_far = math.exp(total_loss / total_tokens)
            print(f"  chunk {i+1}/{n_chunks}  PPL={ppl_so_far:.4f}", flush=True)
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def eval_throughput(model, tok, batch_size, seq_len, n_iters=5, warmup=2):
    from python.throughput_bench import measure_prefill_tok_per_sec

    def fwd(ids):
        return model(ids, use_cache=False).logits

    return measure_prefill_tok_per_sec(
        fwd,
        label=f"qwen36_prefill_b{batch_size}_s{seq_len}",
        vocab_size=tok.vocab_size,
        batch_size=batch_size,
        seq_len=seq_len,
        warmup_iters=warmup,
        n_iters=n_iters,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
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
    print(f"[load] {type(model).__name__} in {time.perf_counter() - t0:.1f}s")

    if args.checkpoint_dir is not None:
        n = install_adapters(model, args.checkpoint_dir, args.mode, args.n_candidates)
        print(f"[swap] adapted {n} layers (mode={args.mode})")
    else:
        print("[swap] skipped — BASELINE")

    result = {
        "model_dir": args.model_dir,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "mode": args.mode if args.checkpoint_dir else "baseline",
        "n_candidates": args.n_candidates if args.mode == "hybrid" else None,
        "eval_mode": args.eval_mode,
    }

    if args.eval_mode == "ppl":
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
