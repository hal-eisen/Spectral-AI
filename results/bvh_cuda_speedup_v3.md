# BVH CUDA-graph fused-projection benchmark (v3)

Closes the gap between `bvh_router_ext`'s 20 µs traversal and the 117–130 µs Python wrapper (Phase 6 v2). The fused path uses `torch.cuda.CUDAGraph` to record the projection→traversal sequence once and replay it with near-zero per-call dispatch overhead.

Measured on RTX 4070 Ti Super (Ada sm_89), `bvh_router_ext` from `cuda/v5/`, 64 experts, BVH 4×4×4. Median of 500 iterations after 20 warmup, CUDA-synced.

| Model | hidden | batch | PyT gate µs | BVH py µs | BVH cuda-graph µs | graph vs py | graph vs PyT gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLMoE-like | 2048 | 1 | 49.29 | 158.9 | **19.21** | 8.27× | **2.57×** |
| OLMoE-like | 2048 | 64 | 52.71 | 138.55 | **26.68** | 5.19× | **1.98×** |
| OLMoE-like | 2048 | 256 | 53.64 | 139.17 | **32.92** | 4.23× | **1.63×** |
| OLMoE-like | 2048 | 1024 | 53.64 | 151.35 | **65.29** | 2.32× | **0.82×** |
| Gemma-4-26B-A4B | 2816 | 1 | 52.43 | 163.0 | **20.09** | 8.11× | **2.61×** |
| Gemma-4-26B-A4B | 2816 | 64 | 53.95 | 140.19 | **27.78** | 5.05× | **1.94×** |
| Gemma-4-26B-A4B | 2816 | 256 | 53.82 | 140.36 | **34.69** | 4.05× | **1.55×** |
| Gemma-4-26B-A4B | 2816 | 1024 | 53.81 | 154.69 | **73.68** | 2.1× | **0.73×** |
| Qwen-3.6-35B-A3B | 2048 | 1 | 50.6 | 160.38 | **18.16** | 8.83× | **2.79×** |
| Qwen-3.6-35B-A3B | 2048 | 64 | 53.38 | 140.11 | **25.23** | 5.55× | **2.12×** |
| Qwen-3.6-35B-A3B | 2048 | 256 | 54.41 | 145.82 | **30.92** | 4.72× | **1.76×** |
| Qwen-3.6-35B-A3B | 2048 | 1024 | 55.52 | 151.8 | **61.25** | 2.48× | **0.91×** |

## What this shows

- The Python BVH wrapper's 117 µs cost was almost entirely Python dispatch overhead: 4 `Linear` ops + a kernel call, each with ~10-20 µs of `torch` dispatch + CUDA launch.
- CUDA graphs collapse all those launches into a single replay, leaving only the kernel work + a small replay overhead.
- This is the practical path to using the BVH primitive without rewriting it in C++. A fused C++/CUDA kernel could go a few µs lower but at much higher engineering cost.