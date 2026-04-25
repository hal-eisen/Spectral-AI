"""Export trained BVH routers to a single mmap-friendly binary file.

Reads checkpoints/{model}_distill_branch_200k/bvh_router_L*_best.pt and writes
a single .bvh file that llama.cpp can mmap.

The file format (little-endian, x86_64):

  [Header, 64 bytes, padded with zeros]
    magic          : char[4]     "BVHR"
    version        : uint32      1
    n_layers       : uint32      number of MoE layers
    input_dim      : uint32      hidden size (2816 for Gemma 4, 2048 for Qwen 3.6)
    n_experts      : uint32      total leaf experts (128 / 256)
    n_level1       : uint32
    n_level2       : uint32
    n_level3       : uint32
    feature_dim    : uint32      128
    spectral_dim   : uint32      64
    flags          : uint32      bit 0 = spectral_mode
    json_offset    : uint64      byte offset to JSON index (relative to file start)
    json_size      : uint64      JSON index size in bytes
    data_offset    : uint64      byte offset to tensor data section
    [pad to 64 bytes]

  [JSON index, json_size bytes]
    A JSON object:
      {
        "layers": [
          {
            "layer_idx": 0,
            "tensors": {
              "input_proj.0.weight": {
                "shape": [512, 2816],
                "dtype": "float32",
                "data_offset": 0,        // byte offset within data section
                "data_size":  5767168
              },
              ...
            },
            "scalars": {
              "temperature": 1.0,
              "topk_accuracy": 0.91,
              ...
            }
          },
          ...
        ]
      }

  [Padding to 64-byte alignment]

  [Data section]
    Concatenated raw tensor bytes; each tensor is 64-byte aligned.

The tensor names mirror the pytorch state_dict (e.g. "level1.centers",
"input_proj.0.weight"). The C++ loader walks the JSON, then for each
(layer, tensor) computes (file_base + data_offset + data_offset_in_section).

Usage:
  python python/export_bvh_router_for_llamacpp.py \
      --checkpoint-dir checkpoints/gemma4_distill_branch_200k \
      --n-layers 30 \
      --out checkpoints/gemma4_bvh_router.bin

  python python/export_bvh_router_for_llamacpp.py \
      --checkpoint-dir checkpoints/qwen36_distill_branch_200k \
      --n-layers 40 \
      --out checkpoints/qwen36_bvh_router.bin
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch

MAGIC = b"BVHR"
VERSION = 1
HEADER_SIZE = 64
ALIGNMENT = 64

# Header layout: magic[4] + 9 uint32 + 3 uint64 = 4 + 36 + 24 = 64 bytes
# (version, n_layers, input_dim, n_experts, n_level1, n_level2, n_level3,
#  feature_dim, spectral_dim+flags-packed, json_offset, json_size, data_offset)
# To avoid overflowing 64 bytes we pack spectral_mode into the high bit of the
# spectral_dim slot (spectral_dim is small, fits in low 16 bits).
HEADER_FMT = "<4sIIIIIIIIIQQQ"
assert struct.calcsize(HEADER_FMT) == HEADER_SIZE, struct.calcsize(HEADER_FMT)


def _round_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


_DTYPE_NAME = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int32: "int32",
    torch.int64: "int64",
}


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    if t.dtype not in _DTYPE_NAME:
        raise ValueError(f"unsupported dtype {t.dtype} for tensor of shape {tuple(t.shape)}")
    return t.detach().contiguous().cpu().numpy().tobytes()


def _scalarize(v):
    """Convert torch scalars / tensors to JSON-serializable Python values."""
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
        return None  # not a scalar; caller should treat as tensor
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    return repr(v)


def export_router(checkpoint_dir: Path, n_layers: int, out_path: Path) -> None:
    # ---- Pass 1: load every layer, collect tensors + per-layer scalars ----
    per_layer = []          # list of dicts: {layer_idx, tensors: {name: tensor}, scalars: {...}, config: {...}}
    common_config = None

    for li in range(n_layers):
        ckpt_path = checkpoint_dir / f"bvh_router_L{li}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["router_state_dict"]
        cfg = ckpt.get("config", {}) or {}
        if common_config is None:
            common_config = cfg
        else:
            for k in ("input_dim", "n_experts", "n_level1", "n_level2", "n_level3",
                      "feature_dim", "spectral_dim", "spectral_mode"):
                if cfg.get(k) != common_config.get(k):
                    raise ValueError(
                        f"layer {li} config[{k}]={cfg.get(k)} disagrees with "
                        f"layer 0 config[{k}]={common_config.get(k)}"
                    )

        scalars = {}
        for k in ("epoch", "router_type", "topk_accuracy", "top1_accuracy",
                  "spectral_mode", "beta"):
            if k in ckpt:
                s = _scalarize(ckpt[k])
                if s is not None:
                    scalars[k] = s

        tensors = {name: t for name, t in sd.items() if isinstance(t, torch.Tensor)}
        per_layer.append({
            "layer_idx": li,
            "tensors": tensors,
            "scalars": scalars,
        })

    if common_config is None:
        raise RuntimeError("no checkpoints loaded -- nothing to export")

    # ---- Pass 2: assign data offsets within the data section ----
    layer_blocks = []
    cursor = 0
    for entry in per_layer:
        index = {}
        for name, t in entry["tensors"].items():
            payload = _tensor_to_bytes(t)
            index[name] = {
                "shape": list(t.shape),
                "dtype": _DTYPE_NAME[t.dtype],
                "data_offset": cursor,
                "data_size": len(payload),
                "_payload": payload,        # internal, dropped before JSON dump
            }
            cursor = _round_up(cursor + len(payload), ALIGNMENT)
        layer_blocks.append({
            "layer_idx": entry["layer_idx"],
            "tensors": index,
            "scalars": entry["scalars"],
        })
    data_section_size = cursor

    # ---- Build the JSON index (strip internal _payload before encoding) ----
    json_obj = {
        "version": VERSION,
        "n_layers": len(layer_blocks),
        "config": {k: common_config.get(k) for k in (
            "input_dim", "n_experts", "n_level1", "n_level2", "n_level3",
            "feature_dim", "spectral_dim", "spectral_mode")},
        "layers": [
            {
                "layer_idx": b["layer_idx"],
                "scalars": b["scalars"],
                "tensors": {
                    name: {k: v for k, v in entry.items() if not k.startswith("_")}
                    for name, entry in b["tensors"].items()
                },
            }
            for b in layer_blocks
        ],
    }
    json_bytes = json.dumps(json_obj, separators=(",", ":")).encode("utf-8")
    json_size = len(json_bytes)

    # ---- Layout: header [0..64) | json [64..64+json_size) | pad | data ----
    json_offset = HEADER_SIZE
    data_offset = _round_up(json_offset + json_size, ALIGNMENT)
    file_size = data_offset + data_section_size

    # ---- Pack header ----
    cfg = common_config
    spectral_dim = int(cfg["spectral_dim"])
    if spectral_dim > 0xFFFF:
        raise ValueError(f"spectral_dim {spectral_dim} doesn't fit in 16 bits")
    spectral_flags = (1 << 31) if bool(cfg.get("spectral_mode", 0)) else 0
    spectral_packed = spectral_flags | spectral_dim
    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        VERSION,
        len(layer_blocks),
        int(cfg["input_dim"]),
        int(cfg["n_experts"]),
        int(cfg["n_level1"]),
        int(cfg["n_level2"]),
        int(cfg["n_level3"]),
        int(cfg["feature_dim"]),
        spectral_packed,
        json_offset,
        json_size,
        data_offset,
    )

    # ---- Write file ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        assert f.tell() == HEADER_SIZE
        f.write(json_bytes)
        # pad to data section
        pad_n = data_offset - f.tell()
        if pad_n > 0:
            f.write(b"\x00" * pad_n)
        assert f.tell() == data_offset

        for block in layer_blocks:
            for name, entry in block["tensors"].items():
                payload = entry["_payload"]
                expected = data_offset + entry["data_offset"]
                cur = f.tell()
                if cur < expected:
                    f.write(b"\x00" * (expected - cur))
                elif cur > expected:
                    raise RuntimeError(
                        f"alignment overrun on {name}: at {cur}, expected {expected}"
                    )
                f.write(payload)

        # Final pad to file_size
        cur = f.tell()
        if cur < file_size:
            f.write(b"\x00" * (file_size - cur))

    print(f"[export] wrote {out_path}")
    print(f"  layers      = {len(layer_blocks)}")
    print(f"  input_dim   = {cfg['input_dim']}")
    print(f"  n_experts   = {cfg['n_experts']}")
    print(f"  tree shape  = {cfg['n_level1']} x {cfg['n_level2']} x {cfg['n_level3']}")
    print(f"  feature_dim = {cfg['feature_dim']}")
    print(f"  spectral    = {cfg['spectral_dim']} (mode={int(bool(cfg.get('spectral_mode',0)))})")
    print(f"  json_size   = {json_size} bytes  (offset {json_offset})")
    print(f"  data_size   = {data_section_size} bytes  (offset {data_offset})")
    print(f"  file_size   = {file_size} bytes "
          f"({file_size/1024/1024:.2f} MB)")
    n_tensors_total = sum(len(b["tensors"]) for b in layer_blocks)
    print(f"  tensors     = {n_tensors_total} ({n_tensors_total // len(layer_blocks)} per layer)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", type=Path, required=True,
                    help="e.g. checkpoints/gemma4_distill_branch_200k")
    ap.add_argument("--n-layers", type=int, required=True,
                    help="Number of MoE layers (Gemma 4 = 30, Qwen 3.6 = 40)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output .bin file path")
    args = ap.parse_args()
    export_router(args.checkpoint_dir, args.n_layers, args.out)


if __name__ == "__main__":
    main()
