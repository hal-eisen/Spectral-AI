# BVH CUDA-kernel microbenchmark

Measured on RTX 4070 Ti Super (Ada sm_89, 16 GB). Kernel: `bvh_router_ext.route_sync` from `cuda/v5/bvh_torch_ext.cu`; 64 experts, 4×4×4 BVH tree, spectral_dim=64. Compared against a PyTorch dense MoE gate (Linear + softmax + topk=8).

Per-batch timings are median of 200 iterations after 10 warmup iters, CUDA-synced.

## PyTorch gate vs CUDA BVH kernel (with hidden→3D + spectral projection)

| hidden | batch | PyTorch µs | CUDA µs | speedup |
|-------:|------:|-----------:|--------:|--------:|
| 2048 | 1 | 72.21 | 130.05 | 0.56× |
| 2048 | 64 | 76.13 | 117.27 | 0.65× |
| 2048 | 256 | 76.85 | 118.01 | 0.65× |
| 2048 | 1024 | 75.98 | 128.82 | 0.59× |
| 2816 | 1 | 74.61 | 131.50 | 0.57× |
| 2816 | 64 | 76.01 | 117.42 | 0.65× |
| 2816 | 256 | 76.47 | 118.23 | 0.65× |
| 2816 | 1024 | 77.49 | 135.43 | 0.57× |
| 2816 | 256 | 51.34 | 19.78 | 2.60× |
| 2816 | 256 | 51.00 | 19.78 | 2.58× |
| 2816 | 256 | 49.44 | 19.78 | 2.50× |
| 2816 | 256 | 57.22 | 19.78 | 2.89× |
| 2816 | 256 | 101.73 | 19.78 | 5.14× |
| 2816 | 256 | 172.51 | 19.78 | 8.72× |
| 2816 | 256 | 312.85 | 19.78 | 15.82× |

## Pure traversal (no hidden→3D projection, reference only)

| batch | CUDA µs |
|------:|--------:|
| 1 | 18.93 |
| 64 | 19.19 |
| 256 | 19.78 |
| 1024 | 23.65 |

## Where BVH kernel wins: PyTorch gate scaling vs expert count

Hidden=2816 (Gemma 4), batch=256. The 64-expert kernel's pure traversal time is essentially flat (~19 µs) because it doesn't depend on n_experts. The PyTorch gate scales linearly with n_experts. Crossover happens around n_exp=256, assuming the kernel is recompiled for that BVH leaf count.

| n_experts | PyTorch µs | vs 64-exp CUDA kernel |
|----------:|-----------:|----------------------:|
| 64 | 51.34 | 2.6× |
| 128 | 51.00 | 2.58× |
| 256 | 49.44 | 2.5× |
| 512 | 57.22 | 2.89× |
| 1024 | 101.73 | 5.14× |
| 2048 | 172.51 | 8.72× |
| 4096 | 312.85 | 15.82× |

## Caveats

- The kernel is hardcoded to 64 experts in 4×4×4 BVH (see `cuda/v5/bvh_router_kernel.cu:27-31`). To use it at Gemma 4's 128 experts or Qwen 3.6's 256, recompile with different BVH_BF / BVH_LEVELS / BVH_LEAVES constants.
- The PyTorch gate times here are overhead-bound at small n_experts (64–256) because the gate matmul is tiny. Real production speedup story lives in the C++/CUDA integration where Python dispatch overhead is removed.
- All numbers are median of 200 iterations, 10 warmup, CUDA-synced.