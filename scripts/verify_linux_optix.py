#!/usr/bin/env python3
"""Verify the OptiX RT-Core and CUDA-kernel BVH paths on native Linux.

Prints a PASS/FAIL per component and writes a JSON summary to
results/rt_core_probe.json so downstream tasks (P1-6, P3-6) know which path
can deliver the headline tokens/sec number.

Components probed, in order:
  1. libnvoptix.so.1 present and loadable (driver runtime, non-stub)
  2. OptiX SDK headers available (optix.h) for compilation
  3. PTX build artifacts (optix_router_raygen.ptx etc.) available
  4. optix_training_ext PyTorch extension loadable
  5. End-to-end OptiX routing on 64 dummy experts (if 1-4 pass)
  6. CUDA-kernel BVH fallback (cuda/v5/bvh_router_kernel.cu) — always on Linux
  7. PyTorch-only gate baseline (sanity)

Exit 0 if *any* GPU-accelerated routing path works. Exit 1 only if we
cannot route at all.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULT: dict = {
    "tests": {},
    "recommended_backend": None,
    "notes": [],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _record(name: str, status: str, **extra):
    entry = {"status": status}
    entry.update(extra)
    RESULT["tests"][name] = entry
    tag = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}.get(status, status.upper())
    print(f"[{tag:4}] {name}" + ("" if not extra else f"  — {extra}"))


# ── 1. libnvoptix.so.1 ────────────────────────────────────────────────
def test_libnvoptix() -> bool:
    lib_name = ctypes.util.find_library("nvoptix") or "libnvoptix.so.1"
    try:
        lib = ctypes.CDLL(lib_name)
    except OSError as e:
        _record("libnvoptix_load", "fail", error=str(e), searched=lib_name)
        return False

    # The driver lib exposes optixQueryFunctionTable. If this symbol is
    # missing, we're looking at a WSL2 stub.
    try:
        _ = lib.optixQueryFunctionTable
        _record("libnvoptix_load", "pass", path=lib_name, has_query_table=True)
        return True
    except AttributeError:
        _record(
            "libnvoptix_load",
            "fail",
            error="optixQueryFunctionTable missing (stub lib?)",
            path=lib_name,
        )
        return False


# ── 2. OptiX SDK headers ──────────────────────────────────────────────
def test_optix_sdk() -> Path | None:
    candidates = [
        os.environ.get("OptiX_INSTALL_DIR"),
        "/opt/nvidia/optix",
        "/usr/local/optix",
        "/usr/local/include/optix",
        str(Path.home() / "NVIDIA-OptiX-SDK"),
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if (p / "include" / "optix.h").exists():
            _record("optix_sdk_headers", "pass", path=str(p))
            return p
        if (p / "optix.h").exists():
            _record("optix_sdk_headers", "pass", path=str(p))
            return p
    _record(
        "optix_sdk_headers",
        "fail",
        error="optix.h not found in any standard location or $OptiX_INSTALL_DIR",
    )
    return None


# ── 3. PTX artifacts ──────────────────────────────────────────────────
def test_ptx_artifacts() -> bool:
    search = [
        PROJECT_ROOT / "build" / "ptx",
        PROJECT_ROOT / "build" / "Release" / "ptx",
        PROJECT_ROOT / "build",
    ]
    raygen = hitgroup = None
    for d in search:
        if not d.exists():
            continue
        for ext in ("*.optixir", "*.ptx"):
            for f in d.glob(ext):
                n = f.stem.lower()
                if "raygen" in n and "router" in n and raygen is None:
                    raygen = f
                if "hitgroup" in n and "router" in n and hitgroup is None:
                    hitgroup = f
    if raygen and hitgroup:
        _record("ptx_artifacts", "pass", raygen=str(raygen), hitgroup=str(hitgroup))
        return True
    _record(
        "ptx_artifacts",
        "fail",
        error="router raygen/hitgroup PTX not found (CMake build not run)",
    )
    return False


# ── 4. optix_training_ext loadable ────────────────────────────────────
def test_optix_ext() -> bool:
    try:
        import optix_training_ext  # type: ignore

        _record("optix_training_ext_import", "pass")
        return True
    except ImportError as e:
        _record("optix_training_ext_import", "fail", error=str(e))
        return False


# ── 5. End-to-end OptiX routing ───────────────────────────────────────
def test_optix_e2e(n_experts: int = 64, batch: int = 256) -> bool:
    try:
        import torch

        from python.optix_training_bridge import OptiXTrainingBridge
    except Exception as e:  # noqa: BLE001
        _record("optix_e2e", "fail", error=f"import error: {e}")
        return False

    try:
        bridge = OptiXTrainingBridge(auto_init=True)
    except Exception as e:  # noqa: BLE001
        _record("optix_e2e", "fail", error=f"bridge init: {e}")
        return False

    if not bridge.has_extension:
        _record("optix_e2e", "skip", error="extension not loaded")
        return False

    dev = torch.device("cuda")
    centers = torch.randn(n_experts, 3).float()
    radii = torch.full((n_experts,), 0.15)

    if not bridge.build_gas(centers, radii):
        _record("optix_e2e", "fail", error="build_gas returned False")
        return False

    pos = torch.randn(batch, 3, device=dev)
    dirs = torch.randn(batch, 3, device=dev)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    # Warmup + timed
    for _ in range(5):
        _ = bridge.route_rt(pos, dirs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        _ = bridge.route_rt(pos, dirs)
    torch.cuda.synchronize()
    us_per = (time.perf_counter() - t0) * 1e6 / 50

    _record("optix_e2e", "pass", us_per_batch=round(us_per, 2), n_experts=n_experts, batch=batch)
    return True


# ── 6. CUDA-kernel BVH fallback ───────────────────────────────────────
def test_cuda_kernel_bvh(n_experts: int = 64, batch: int = 256) -> bool:
    """Benchmark the compiled v5 CUDA kernel (bvh_router_ext) directly.

    The Python-level BranchSpecificBVHRouter is 30x slower than the gate because
    of per-call Python overhead (spectral encoder, portals, gumbel). The headline
    "fast" path is the compiled kernel — separately built via cuda/v5/build_ext.py.
    """
    import torch

    ext_dir = os.path.expanduser("~/.cache/torch_extensions/bvh_router_ext")
    if not Path(ext_dir, "bvh_router_ext.so").exists():
        _record(
            "cuda_kernel_bvh",
            "fail",
            error=f"bvh_router_ext.so not built; run `python cuda/v5/build_ext.py`",
        )
        return False

    sys.path.insert(0, ext_dir)
    try:
        import bvh_router_ext as ext
    except Exception as e:  # noqa: BLE001
        _record("cuda_kernel_bvh", "fail", error=f"import: {e}")
        return False

    dev = torch.device("cuda")

    # Build a 4×4×4 = 64-expert BVH with identity portals and spectral layer
    centers = torch.randn(n_experts, 3).float()
    radii = torch.full((n_experts,), 0.15)
    portals = torch.zeros(21, 3, 4)
    for i in range(21):
        portals[i, 0, 0] = portals[i, 1, 1] = portals[i, 2, 2] = 1.0
    spectral_dim = 64
    snell_w = (
        torch.eye(spectral_dim).unsqueeze(0).expand(n_experts, spectral_dim, spectral_dim).contiguous()
    )
    snell_b = torch.zeros(n_experts, spectral_dim)
    try:
        ext.upload_tree(centers, radii, portals, snell_w, snell_b)
    except Exception as e:  # noqa: BLE001
        _record("cuda_kernel_bvh", "fail", error=f"upload_tree: {e}")
        return False

    results_by_batch = {}
    for b in (batch, min(batch * 4, 1024)):
        origins = torch.randn(b, 3, device=dev).contiguous()
        dirs = torch.randn(b, 3, device=dev)
        dirs = (dirs / dirs.norm(dim=-1, keepdim=True)).contiguous()
        spectral = torch.randn(b, spectral_dim, device=dev).contiguous()

        # Warmup
        for _ in range(10):
            _ = ext.route_sync(origins, dirs, spectral)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_iter = 200
        for _ in range(n_iter):
            _ = ext.route_sync(origins, dirs, spectral)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) * 1e6 / n_iter
        results_by_batch[f"batch_{b}"] = round(us, 2)

    _record(
        "cuda_kernel_bvh",
        "pass",
        us_per_batch=results_by_batch[f"batch_{batch}"],
        all_batches_us=results_by_batch,
        n_experts=n_experts,
        source="compiled bvh_router_ext (cuda/v5/bvh_torch_ext.cu)",
    )
    return True


# ── 7. PyTorch-only gate baseline ─────────────────────────────────────
def test_pytorch_baseline(n_experts: int = 64, batch: int = 256) -> bool:
    try:
        import torch
        import torch.nn as nn
    except Exception as e:  # noqa: BLE001
        _record("pytorch_gate_baseline", "fail", error=str(e))
        return False

    dev = torch.device("cuda")
    gate = nn.Linear(2048, n_experts, bias=False).to(dev).eval()
    x = torch.randn(batch, 2048, device=dev)
    with torch.no_grad():
        for _ in range(5):
            _ = gate(x).softmax(dim=-1).topk(8)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            _ = gate(x).softmax(dim=-1).topk(8)
    torch.cuda.synchronize()
    us_per = (time.perf_counter() - t0) * 1e6 / 50
    _record("pytorch_gate_baseline", "pass", us_per_batch=round(us_per, 2), batch=batch)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "rt_core_probe.json")
    args = ap.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))

    # Non-CUDA prerequisites first
    has_runtime = test_libnvoptix()
    sdk_path = test_optix_sdk()
    has_ptx = test_ptx_artifacts()
    has_ext = test_optix_ext()

    # The OptiX e2e path requires ALL of {runtime, SDK or prebuilt ext, PTX}
    if has_runtime and has_ext and has_ptx:
        optix_ok = test_optix_e2e()
    else:
        optix_ok = False
        _record("optix_e2e", "skip", error="prerequisites not met")

    # CUDA-kernel fallback should always work on this box
    cuda_kernel_ok = test_cuda_kernel_bvh()

    # Baseline for comparison
    baseline_ok = test_pytorch_baseline()

    # Decide recommended backend
    if optix_ok:
        RESULT["recommended_backend"] = "optix_rt"
    elif cuda_kernel_ok:
        RESULT["recommended_backend"] = "cuda_kernel_bvh"
    else:
        RESULT["recommended_backend"] = "pytorch_only" if baseline_ok else "none"

    RESULT["notes"].append(
        f"libnvoptix runtime: {'OK' if has_runtime else 'MISSING/STUB'}. "
        f"OptiX SDK headers: {'at ' + str(sdk_path) if sdk_path else 'NOT INSTALLED'}. "
        f"PTX artifacts: {'found' if has_ptx else 'missing (run CMake build)'}. "
        f"Extension loaded: {has_ext}."
    )
    if not sdk_path:
        RESULT["notes"].append(
            "To enable OptiX RT-core path: download NVIDIA OptiX SDK 9.1+ from "
            "https://developer.nvidia.com/optix/downloads and export "
            "OptiX_INSTALL_DIR=/path/to/SDK before running CMake."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(RESULT, indent=2, default=str))

    print()
    print("=" * 60)
    print(f"Recommended backend: {RESULT['recommended_backend']}")
    for n in RESULT["notes"]:
        print(f"  • {n}")
    print(f"Wrote {args.out}")

    # Exit 0 if we can route at all, else 1
    any_path = optix_ok or cuda_kernel_ok or baseline_ok
    sys.exit(0 if any_path else 1)


if __name__ == "__main__":
    main()
