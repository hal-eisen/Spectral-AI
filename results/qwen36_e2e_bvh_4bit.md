# Qwen 3.6 35B A3B — e2e tok/s with experts at 4-bit on GPU

Config: experts NF4-quantized on GPU for first 20 of 40 layers, remaining 20 layers' experts on CPU bf16. Dense parts (linear/self attention, routers, shared experts, embed, lm_head) on GPU bf16. Total VRAM ~12.21 GB on RTX 4070 Ti Super (16 GB).

Prefill (B=1, S=512), median of 5 runs:

| Config | Median latency (ms) | Tok/s | Speedup |
|---|---:|---:|---:|
| baseline (original gate) | 5136 | 99.69 | 1.00× |
| BVH hybrid n_cand=128 | 5186 | 98.72 | **0.990×** |