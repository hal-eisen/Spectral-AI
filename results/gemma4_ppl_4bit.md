# Gemma 4 26B A4B — PPL fidelity with 4-bit experts on GPU

Same WikiText-2 chunks (50, ctx=512) as the full-bf16 baseline. Tests whether the 4-bit-experts deployment configuration retains fidelity to the original safetensor model, with and without BVH routing on top.

| Config | PPL | Δ vs full-bf16 baseline |
|---|---:|---:|
| Full bf16, original gate | 11.8725 | baseline |
| Full bf16, BVH hybrid n=64 (200k) | 11.9380 | +0.55% |
| **4-bit experts, original gate** | **12.1144** | **+2.04%** |
| **4-bit experts, BVH hybrid n=64 (200k)** | **12.1003** | **+1.92%** |

## Summary

- 4-bit experts costs **2.04%** PPL vs the full-bf16 baseline (typical for NF4 quantization on MoE).
- Adding BVH on top of 4-bit experts gives an additional **-0.12%** PPL — same routing-quality trade-off we saw on the full-bf16 model.
- Combined: 4-bit experts + BVH n=64 is **+1.92%** vs the original safetensor baseline. The total degradation is dominated by 4-bit quantization, not by BVH routing.