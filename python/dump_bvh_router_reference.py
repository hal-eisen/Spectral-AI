"""Dump deterministic reference (input, output) tensors for the C++ ggml-graph
BVH router validator (P782-3 / P782-3b).

Subcommands:
  dump          : dump input + input_proj output (256-dim)
  dump-full     : dump input + full router logits (n_experts) + expert_ids
  compare       : compare two float32 .bin files, report max-abs delta
  compare-ids   : compare two int32 .bin files, report agreement rate

Usage:
  python python/dump_bvh_router_reference.py dump-full \
      --checkpoint checkpoints/gemma4_distill_branch_200k/bvh_router_L0_best.pt \
      --batch 4 \
      --out-input    /tmp/bvh_input.bin \
      --out-logits   /tmp/bvh_expected_logits.bin \
      --out-ids      /tmp/bvh_expected_ids.bin
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import torch
import torch.nn as nn


def build_input_proj(state_dict: dict, input_dim: int) -> nn.Sequential:
    """Mirror EnhancedBVHRouter.input_proj from python/olmoe_bvh_distill.py:
        Linear(input_dim, 512) -> GELU -> Linear(512, 256) -> LayerNorm(256)"""
    proj = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.GELU(),
        nn.Linear(512, 256),
        nn.LayerNorm(256),
    )
    proj[0].weight.data.copy_(state_dict["input_proj.0.weight"])
    proj[0].bias.data.copy_(state_dict["input_proj.0.bias"])
    proj[2].weight.data.copy_(state_dict["input_proj.2.weight"])
    proj[2].bias.data.copy_(state_dict["input_proj.2.bias"])
    proj[3].weight.data.copy_(state_dict["input_proj.3.weight"])
    proj[3].bias.data.copy_(state_dict["input_proj.3.bias"])
    proj.eval()
    return proj


def deterministic_input(batch: int, input_dim: int, seed: int = 0xC0FFEE) -> torch.Tensor:
    """Use torch's RNG (NOT C++ mt19937) so the input file is canonical and
    both languages just read it from disk."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.empty(batch, input_dim).uniform_(-0.5, 0.5, generator=g)


def cmd_dump(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["router_state_dict"]
    input_dim = ckpt["config"]["input_dim"]

    proj = build_input_proj(sd, input_dim)

    x = deterministic_input(args.batch, input_dim, args.seed)
    with torch.no_grad():
        h = proj(x)

    args.out_input.parent.mkdir(parents=True, exist_ok=True)
    args.out_input.write_bytes(x.float().contiguous().numpy().tobytes())
    args.out_output.write_bytes(h.float().contiguous().numpy().tobytes())

    print(f"  input_dim       = {input_dim}")
    print(f"  batch           = {args.batch}  seed={args.seed}")
    print(f"  wrote input    -> {args.out_input}  ({x.numel() * 4} bytes)")
    print(f"  wrote expected -> {args.out_output} ({h.numel() * 4} bytes)")
    print(f"  expected[0, :8] = {h[0, :8].tolist()}")
    print(f"  expected[{args.batch-1}, :8] = {h[args.batch-1, :8].tolist()}")
    print(f"  range           = [{h.min().item():.4f}, {h.max().item():.4f}]")


def cmd_compare(args):
    expected = torch.frombuffer(args.expected.read_bytes(), dtype=torch.float32)
    actual   = torch.frombuffer(args.actual.read_bytes(), dtype=torch.float32)
    if expected.numel() != actual.numel():
        print(f"shape mismatch: expected {expected.numel()} floats, "
              f"got {actual.numel()}")
        sys.exit(1)
    delta = (actual - expected).abs()
    rel = delta / (expected.abs() + 1e-6)
    print(f"  n_elements    = {expected.numel()}")
    print(f"  max abs delta = {delta.max().item():.6e}")
    print(f"  mean abs delta= {delta.mean().item():.6e}")
    print(f"  max rel delta = {rel.max().item():.6e}")
    print(f"  expected[0:8] = {expected[:8].tolist()}")
    print(f"  actual[0:8]   = {actual[:8].tolist()}")
    if delta.max().item() < args.tol:
        print(f"  PASS (within tol={args.tol})")
    else:
        print(f"  FAIL (tol={args.tol})")
        sys.exit(2)


def cmd_dump_full(args):
    """Build the full EnhancedBVHRouter from checkpoint and dump logits + ids."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from olmoe_bvh_distill import EnhancedBVHRouter
    import spectral_techniques as st

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    sd  = ckpt["router_state_dict"]
    beta = ckpt.get("beta", 1.0)

    router = EnhancedBVHRouter(
        input_dim   = cfg["input_dim"],
        n_level1    = cfg["n_level1"],
        n_level2    = cfg["n_level2"],
        n_level3    = cfg["n_level3"],
        feature_dim = cfg["feature_dim"],
        spectral_mode = cfg["spectral_mode"],
        spectral_dim  = cfg["spectral_dim"],
    )
    router.load_state_dict(sd)
    router.eval()
    st.set_ste_beta(beta)

    x = deterministic_input(args.batch, cfg["input_dim"], args.seed)
    with torch.no_grad():
        # The forward returns (probs, ids); we want pre-softmax logits, which
        # the router stores at self._last_logits (post-RMSNorm).
        probs, ids = router(x)
        logits = router._last_logits

    args.out_input.parent.mkdir(parents=True, exist_ok=True)
    args.out_input .write_bytes(x.float().contiguous().numpy().tobytes())
    args.out_logits.write_bytes(logits.float().contiguous().numpy().tobytes())
    args.out_ids   .write_bytes(ids.to(torch.int32).contiguous().numpy().tobytes())

    print(f"  input_dim   = {cfg['input_dim']}")
    print(f"  n_experts   = {cfg['n_experts']}")
    print(f"  beta (eval) = {beta:.3f}")
    print(f"  batch       = {args.batch}  seed={args.seed}")
    print(f"  expert_ids  = {ids.tolist()}")
    print(f"  logits[0,:8]= {logits[0,:8].tolist()}")
    print(f"  logits range= [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"  wrote input  -> {args.out_input}  ({x.numel() * 4} bytes)")
    print(f"  wrote logits -> {args.out_logits}")
    print(f"  wrote ids    -> {args.out_ids}")


def cmd_compare_ids(args):
    expected = torch.frombuffer(args.expected.read_bytes(), dtype=torch.int32)
    actual   = torch.frombuffer(args.actual.read_bytes(),   dtype=torch.int32)
    n = min(expected.numel(), actual.numel())
    agree = (expected[:n] == actual[:n]).sum().item()
    print(f"  n_samples = {n}")
    print(f"  agreement = {agree}/{n}  ({100.0*agree/n:.2f}%)")
    print(f"  expected  = {expected[:min(16, n)].tolist()}")
    print(f"  actual    = {actual[:min(16, n)].tolist()}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True, dest="cmd")

    d = sub.add_parser("dump", help="Generate deterministic input + PyTorch input_proj output")
    d.add_argument("--checkpoint", type=Path, required=True)
    d.add_argument("--batch", type=int, default=4)
    d.add_argument("--seed", type=lambda s: int(s, 0), default=0xC0FFEE)
    d.add_argument("--out-input",  type=Path, required=True)
    d.add_argument("--out-output", type=Path, required=True)
    d.set_defaults(func=cmd_dump)

    df = sub.add_parser("dump-full", help="Generate input + full-router logits + expert_ids")
    df.add_argument("--checkpoint", type=Path, required=True)
    df.add_argument("--batch", type=int, default=4)
    df.add_argument("--seed", type=lambda s: int(s, 0), default=0xC0FFEE)
    df.add_argument("--out-input",  type=Path, required=True)
    df.add_argument("--out-logits", type=Path, required=True)
    df.add_argument("--out-ids",    type=Path, required=True)
    df.set_defaults(func=cmd_dump_full)

    c = sub.add_parser("compare", help="Compare expected vs actual float32 output")
    c.add_argument("--expected", type=Path, required=True)
    c.add_argument("--actual",   type=Path, required=True)
    c.add_argument("--tol",      type=float, default=1e-3)
    c.set_defaults(func=cmd_compare)

    ci = sub.add_parser("compare-ids", help="Compare two int32 expert_ids dumps")
    ci.add_argument("--expected", type=Path, required=True)
    ci.add_argument("--actual",   type=Path, required=True)
    ci.set_defaults(func=cmd_compare_ids)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
