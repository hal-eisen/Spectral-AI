# Tokens/sec — Baseline vs SpectralAI BVH routing

Measured on **RTX 4070 Ti Super (Ada sm_89, 16 GB)**, native Linux, CUDA 12.4,
PyTorch 2.10.0+cu128. Both models loaded in bfloat16 with accelerate auto-spill
(max_memory = 12 GiB GPU, rest CPU). Prefill-only (batch=1, seq_len=512),
median of 5 runs after 2 warmup runs, CUDA-synced timing.

The BVH path in this PyTorch harness runs **both** the original router (to
trigger accelerate's pre-forward hook that loads offloaded parameters) AND the
BVH router for candidate selection. So the "with SpectralAI" numbers below
reflect BVH overhead + original-router cost — they do NOT show a speedup.
The real speedup claim is at the C++/CUDA kernel level (Phase 6, deferred):
bvh_router_ext on Linux sm_89 hits 19.13 µs/batch=256, already matching the
published Windows+OptiX 19.1 µs number (see `results/rt_core_probe.json`).

## Models under test

| Model | Total params | Active | Experts | Hidden | Layers |
|---|---:|---:|---:|---:|---:|
| Gemma 4 26B A4B | 26 B | 4 B | 128 (top-8) | 2816 | 30 |
| Qwen 3.6 35B A3B | 35 B | 3 B | 256 (top-8) | 2048 | 40 |

## Prefill tok/s (batch=1, seq=512, PyTorch bf16)

| Model | Baseline tok/s | BVH hybrid tok/s | Δ |
|---|---:|---:|---:|
| Gemma 4 26B A4B | **107.19 ± 0.20** | **106.06 ± 0.07** | −1.1% |
| Qwen 3.6 35B A3B | **72.82 ± 0.56** | **72.09 ± 0.18** | −1.0% |

(BVH hybrid uses n_candidates=32 in all runs.)

## For reference: llama.cpp baselines (same hardware, different backend)

llama.cpp with CUDA 12.4, `llama-bench -p 512 -n 128 -r 3`, prefill tok/s. See
`results/olmoe_llamacpp_baseline.csv` and `results/qwen36_llamacpp.csv`.

| Model | Quant | Size GB | Prefill tok/s | Generate tok/s |
|---|---|---:|---:|---:|
| OLMoE 7B-A1B | Q4_K_M | 3.92 | 14 675 | 524 |
| OLMoE 7B-A1B | F16 | 12.89 | 6 620 | 212 |
| Qwen 3.6 35B-A3B | UD-Q4_K_S | 19.45 | 447 | 37 |

The llama.cpp numbers dwarf the PyTorch numbers because llama.cpp uses fused
CUDA kernels and no accelerate coupling. These are the numbers the Phase 6
C++/CUDA BVH integration would compete with.

## Notes + caveats

- Gemma 4 throughput stable across 3 back-to-back measurement runs (99.3 →
  107.19 → 107.46 tok/s — variance was due to Qwen training sharing the GPU).
- Qwen 3.6 throughput stable: 72.82 baseline → 72.09 with BVH, single run each.
- The `−1%` overhead of BVH is essentially free at PyTorch level; given the
  routing primitive lands in ~19 µs on the compiled kernel, the real-world
  impact depends on how much of the forward pass is router vs expert FFN.

## Reproduction

```bash
# Both models, all 4 configs each (takes ~45 min per model)
scripts/run_e2e_evals.sh gemma4
scripts/run_e2e_evals.sh qwen36

# Results land in results/{gemma4,qwen36}_e2e.json
```

Prerequisites:
1. Safetensors mounted under `/home/eisen/spectralai/remote_models/` (sshfs OK).
2. BVH router checkpoints in `checkpoints/{gemma4,qwen36}_distill_branch/`
   (produced by `scripts/train_all_layers.sh`, requires hidden-state extraction
   via `python/extract_router_io.py` first).
