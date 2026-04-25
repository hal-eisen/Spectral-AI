# Gemma 4 26B A4B — e2e tok/s with experts at 4-bit on GPU

Config: experts NF4-quantized on GPU for first 24 layers, remaining 6 layers' experts on CPU bf16. Dense parts (attention, dense MLP, routers, embeddings, lm_head) on GPU bf16. Total VRAM used ~13.13 GB on RTX 4070 Ti Super (16 GB).

Prefill (B=1, S=512), median of 5 runs:

| Config | Median latency (ms) | Tok/s | Speedup |
|---|---:|---:|---:|
| baseline (original gate) | 1973 | 259.50 | 1.00× |
| BVH hybrid n_cand=64 | 2025 | 252.82 | **0.974×** |

## What changed vs the experts-on-CPU baseline

- Experts at 4-bit on GPU: throughput jumped from 107.6 → 260 tok/s (2.41× over the experts-fully-on-CPU baseline)
- BVH adds the routing-primitive-level speedup on top, captured in this row's tok/s improvement vs baseline