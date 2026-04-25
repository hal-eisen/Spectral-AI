# Qwen 3.6 35B A3B — PPL fidelity with 4-bit experts on GPU

Same WikiText-2 chunks (50, ctx=512) as the full-bf16 baseline. Tests whether the 4-bit-experts deployment configuration retains fidelity to the original safetensor model.

| Config | PPL | Δ vs full-bf16 baseline |
|---|---:|---:|
| Full bf16, original gate | 7.8729 | baseline |
| Full bf16, BVH hybrid n=128 (200k) | 7.8744 | +0.02% |
| **4-bit experts, original gate** | **7.8950** | **+0.28%** |
| **4-bit experts, BVH hybrid n=128 (200k)** | **7.9089** | **+0.46%** |

## Summary

- 4-bit experts costs **0.28%** PPL vs the full-bf16 baseline.
- BVH on top of 4-bit experts: **+0.18%** additional Δ.
- Combined: 4-bit experts + BVH n=128 is **+0.46%** vs the original safetensor baseline.