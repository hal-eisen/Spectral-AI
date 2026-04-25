# Gemma 4 26B A4B — real e2e tok/s with BVH routing

Surgical device_map: dense layers + embeddings + lm_head + routers on GPU (~4.5 GB VRAM), expert FFNs on CPU (PCIe-loaded on dispatch). This is the natural pattern for fitting Gemma 4 26B on a 16 GB GPU.

Prefill (B=1, S=512), bf16, RTX 4070 Ti Super:

| Config | Median latency (ms) | Tok/s | Speedup |
|---|---:|---:|---:|
| baseline (original gate) | 4755 | 107.68 | 1.00× |
| BVH hybrid n_cand=64 | 4758 | 107.60 | 0.999× |

## What this shows

Both configs share the same hot path: every token activates 8 of 128 experts; those experts' weights are pulled from CPU RAM over PCIe on each forward. Routing primitive cost (Linear[H→E] + softmax + topk OR BVH) is **O(microseconds)**; expert dispatch is **O(milliseconds)**.
BVH cuda-graph microbenchmark (`results/bvh_cuda_speedup_v3.md`) shows the routing primitive is **2-3× faster** with BVH+CUDA graphs than PyTorch's dense gate. That win is real but invisible at the model-tok/s level when the expert PCIe traffic dominates.
Where BVH matters at the model level:
- Experts fully on GPU (no PCIe bottleneck): need 80+ GB or aggressive 3-bit quant
- Models with 1000+ experts (e.g. DeepSeek V3): routing matmul becomes the bottleneck
- Lower-batch settings where the dense gate's launch overhead dominates