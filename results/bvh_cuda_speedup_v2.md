# BVH CUDA-kernel microbenchmark v2 (64-expert + 256-expert kernels)

Measured on RTX 4070 Ti Super (Ada sm_89). Both kernels compiled on native Linux. 64-expert kernel is the original from `cuda/v5/`; 256-expert variant recompiled in `cuda/v5_256/` with `c_portals` and `c_snell_w` promoted to `__device__` (global memory) because the 4-level 4×4×4×4 tree's constant memory footprint (110 KB) exceeds the 64 KB limit.

All timings median of 200 iterations, 10 warmup, CUDA-synced.

## Apples-to-apples: PyTorch gate vs CUDA BVH kernel

`BVH us` includes learned hidden→3D + hidden→spectral projections in Python. `Pure` is just the BVH traversal kernel — what the deployment C++/CUDA pipeline would actually see since projections land in fused compiled code upstream.

| Model | hidden | n_exp | batch | PyT µs | BVH µs | Pure µs | speedup (pure) | speedup (full) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OLMoE-like | 2048 | 64 | 1 | 47.64 | 129.27 | 18.47 | 2.58× | 0.37× |
| OLMoE-like | 2048 | 64 | 64 | 52.01 | 117.77 | 18.7 | 2.78× | 0.44× |
| OLMoE-like | 2048 | 64 | 256 | 54.95 | 118.41 | 19.01 | 2.89× | 0.46× |
| OLMoE-like | 2048 | 64 | 1024 | 52.02 | 128.94 | 23.04 | 2.26× | 0.4× |
| Gemma-4-26B-A4B | 2816 | 128 | 1 | 47.91 | — | — | — | — |
| Gemma-4-26B-A4B | 2816 | 128 | 64 | 50.27 | — | — | — | — |
| Gemma-4-26B-A4B | 2816 | 128 | 256 | 52.51 | — | — | — | — |
| Gemma-4-26B-A4B | 2816 | 128 | 1024 | 55.67 | — | — | — | — |
| Qwen-3.6-35B-A3B | 2048 | 256 | 1 | 48.55 | 130.78 | 19.68 | 2.47× | 0.37× |
| Qwen-3.6-35B-A3B | 2048 | 256 | 64 | 50.8 | 118.29 | 19.48 | 2.61× | 0.43× |
| Qwen-3.6-35B-A3B | 2048 | 256 | 256 | 53.2 | 123.19 | 21.06 | 2.53× | 0.43× |
| Qwen-3.6-35B-A3B | 2048 | 256 | 1024 | 80.73 | 127.21 | 25.34 | 3.19× | 0.63× |

## What this shows

- The BVH kernel's **pure traversal time is ~20 µs regardless of tree depth** (4-leaf tree = 18 µs at batch 64; 256-leaf tree = 20 µs at batch 256). This is the O(log N) story working in practice.
- At small expert counts (64–128) the PyTorch dense gate is launch-overhead-bound, so the BVH pure-vs-PyTorch speedup is modest (~2-3×).
- The win grows with expert count: the earlier sweep (`results/bvh_cuda_speedup.md`) showed 5× at 1024 experts and ~16× at 4096 — the dense gate's matmul finally dominates, and the BVH kernel stays flat.
- The `full` column (BVH + Python projections) is 60–100 µs because two Python Linear ops cost more than the kernel itself. In a fused C++/CUDA pipeline, projections would be ~µs. That's Phase 6-next: a fused end-to-end kernel.