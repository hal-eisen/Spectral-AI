#!/usr/bin/env python3
"""Dump GGUF metadata we care about for BVH routing planning.

Reports hidden_size, n_layers, n_experts, expert-routing config, RoPE, vocab size,
plus any MoE-specific keys. Writes a JSON summary next to the GGUF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gguf


INTERESTING_SUFFIXES = (
    ".architecture",
    ".context_length",
    ".embedding_length",
    ".feed_forward_length",
    ".block_count",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.layer_norm_rms_epsilon",
    ".rope.dimension_count",
    ".rope.freq_base",
    ".rope.scaling.type",
    ".rope.scaling.factor",
    ".expert_count",
    ".expert_used_count",
    ".expert_shared_count",
    ".expert_feed_forward_length",
    ".expert_gating_func",
    ".vocab_size",
    "tokenizer.ggml.model",
    "general.name",
    "general.quantization_version",
    "general.file_type",
)


def inspect(gguf_path: Path) -> dict:
    reader = gguf.GGUFReader(str(gguf_path), "r")

    # Pull architecture first so we can filter prefixed keys sensibly.
    arch = None
    for f in reader.fields.values():
        if f.name == "general.architecture":
            arch = bytes(f.parts[f.data[0]]).decode("utf-8", errors="replace")
            break

    summary: dict = {"path": str(gguf_path), "architecture": arch, "fields": {}}

    for f in reader.fields.values():
        key = f.name
        keep = any(key.endswith(sfx) for sfx in INTERESTING_SUFFIXES) or key.startswith(
            "general."
        )
        if not keep:
            continue
        try:
            if not f.data:
                val = None
            elif len(f.data) == 1 and f.types and f.types[0] == gguf.GGUFValueType.STRING:
                val = bytes(f.parts[f.data[0]]).decode("utf-8", errors="replace")
            else:
                vals = []
                for di in f.data:
                    part = f.parts[di]
                    if part.dtype.kind in ("i", "u"):
                        # int/uint arrays: if dtype is uint8, it's probably a UTF-8 string
                        if part.dtype == "uint8" and part.size > 1:
                            try:
                                vals.append(bytes(part).decode("utf-8", errors="replace"))
                                continue
                            except Exception:
                                pass
                        vals.append(part.tolist() if part.size > 1 else part.item())
                    elif part.dtype.kind == "f":
                        vals.append(part.tolist() if part.size > 1 else part.item())
                    else:
                        try:
                            vals.append(bytes(part).decode("utf-8", errors="replace"))
                        except Exception:
                            vals.append(repr(part))
                val = vals[0] if len(vals) == 1 else vals
        except Exception as e:  # noqa: BLE001
            val = f"<unreadable: {e}>"
        summary["fields"][key] = val

    # Count tensors and categorize MoE-relevant ones.
    moe_tensor_names = []
    gate_tensor_names = []
    for t in reader.tensors:
        name = t.name
        lowered = name.lower()
        if "expert" in lowered or ".ffn_gate_exps" in lowered or ".ffn_up_exps" in lowered:
            moe_tensor_names.append(name)
        if ".ffn_gate_inp" in lowered or ".gate." in lowered:
            gate_tensor_names.append(name)

    summary["n_tensors"] = len(reader.tensors)
    summary["sample_moe_tensors"] = moe_tensor_names[:8]
    summary["sample_gate_tensors"] = gate_tensor_names[:8]

    # Derive the MoE layer indices from the gate-input tensor names (format: blk.N.ffn_gate_inp.weight).
    moe_layer_indices = set()
    for name in gate_tensor_names:
        parts = name.split(".")
        for a, b in zip(parts, parts[1:]):
            if a == "blk" and b.isdigit():
                moe_layer_indices.add(int(b))
    summary["moe_layer_indices"] = sorted(moe_layer_indices)
    summary["n_moe_layers_detected"] = len(moe_layer_indices)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="Write JSON here (default: alongside GGUF)")
    args = ap.parse_args()

    if not args.gguf.exists():
        raise SystemExit(f"GGUF not found: {args.gguf}")

    summary = inspect(args.gguf)

    out = args.out or args.gguf.with_suffix(".arch.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))

    print(f"architecture: {summary['architecture']}")
    print(f"n_tensors: {summary['n_tensors']}")
    print(f"detected MoE layers: {summary['n_moe_layers_detected']}")
    print(f"moe layer indices: {summary['moe_layer_indices']}")
    print("fields:")
    for k, v in summary["fields"].items():
        if isinstance(v, list) and len(v) > 8:
            shown = f"[len={len(v)}] {v[:4]}..."
        else:
            shown = v
        print(f"  {k}: {shown}")
    print(f"sample MoE tensors: {summary['sample_moe_tensors']}")
    print(f"sample gate tensors: {summary['sample_gate_tensors']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
