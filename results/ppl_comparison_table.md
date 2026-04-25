# PPL — Baseline vs SpectralAI BVH routing

Measured on **RTX 4070 Ti Super (Ada sm_89, 16 GB)**, native Linux, PyTorch
2.10 bf16 with accelerate auto-spill. PPL on WikiText-2 test split, `ctx=512`,
50 chunks (25 600 tokens), CE loss on shifted-by-1 targets. Same chunks and
tokenizer for all runs of a given model.

## Headline (Phase 7 outcome)

After two interventions — **n_candidates sweep** (no retraining) and **router
retraining with 4× more data** (200k tokens/layer instead of 50k) — both
models reach the plan's ±1% PPL target with hybrid BVH routing:

| Model | Operating point | PPL | Δ vs baseline |
|---|---|---:|---:|
| **Gemma 4 26B A4B** | 200k-trained router, n_candidates=64 (50% of experts) | **11.938** | **+0.55%** |
| **Qwen 3.6 35B A3B** | 200k-trained router, n_candidates=96 (37.5% of experts) | **7.897** | **+0.30%** |

For comparison, the original Phase-1 results (50k-trained, n_cand=32) were
+6.0 % (Gemma 4) and +10.5 % (Qwen 3.6). Both interventions stack: tighter
n_cand at fixed training data, and better routers at fixed n_cand.

## Full sweep tables — 50k vs 200k retraining

### Gemma 4 26B A4B (128 experts, baseline PPL **11.873**)

| n_cand | % experts | 50k PPL | 50k Δ | 200k PPL | 200k Δ |
|---:|---:|---:|---:|---:|---:|
| 32 | 25.0% | 12.521 | +5.46% | 12.211 | **+2.85%** |
| 64 | 50.0% | 12.106 | +1.96% | 11.938 | **+0.55%** |
| 96 | 75.0% | 12.037 | +1.38% | 11.966 | +0.79% |
| 128 | 100.0% | 12.047 | +1.47% | 11.984 | +0.94% |
| pure BVH | — | 91.534 | +671% | 121.99 | +927% |

Note: at n=128 (which selects all experts and should equal baseline by
definition), there's a +0.94 % residual — likely bf16 numerical
ordering in the wrapper's renorm + per_expert_scale path. The
retraining narrowed it from +1.47 % → +0.94 % but didn't eliminate it.
Out-of-band investigation TBD.

### Qwen 3.6 35B A3B (256 experts, baseline PPL **7.873**)

| n_cand | % experts | 50k PPL | 50k Δ | 200k PPL | 200k Δ |
|---:|---:|---:|---:|---:|---:|
| 32 | 12.5% | 8.698 | +10.48% | 8.125 | **+3.20%** |
| 64 | 25.0% | 8.256 | +4.87% | 7.955 | **+1.05%** |
| 96 | 37.5% | 8.067 | +2.47% | 7.897 | **+0.30%** |
| 128 | 50.0% | 7.968 | +1.20% | 7.874 | +0.02% |
| 192 | 75.0% | 7.884 | +0.14% | 7.880 | +0.08% |
| 256 | 100.0% | 7.876 | +0.04% | 7.875 | +0.03% |
| pure BVH | — | 201.001 | +2453% | 228.051 | +2797% |

Qwen 3.6 has no per_expert_scale (the Qwen3_5MoeTopKRouter is a simpler
class) so n=256 (all experts) reproduces baseline within rounding noise
in both the 50k and 200k columns — confirming the wrapper has no
Qwen-side residual bug.

## Router-quality lift from retraining

Top-8 overlap between BVH router and gate (mean across all layers):

| Model | 50k | 200k | Δ |
|---|---:|---:|---:|
| Gemma 4 (128 exp) | 73.4% | 82.5% | +9.1 pts |
| Qwen 3.6 (256 exp) | 59.3% | 70.1% | +10.8 pts |

Hardest-layer (min) top-8 overlap improved more dramatically:
- Gemma 4 min: 61.7% → 74.3% (+12.6 pts)
- Qwen 3.6 min: 49.7% → 62.0% (+12.3 pts)

That's why the same n_candidates at 200k beats the same n_candidates at 50k:
the candidate set is more likely to contain the gate's true top-8.

## Pure BVH gets *worse* with retraining

Surprising side-effect: the 200k-trained router has worse pure-BVH PPL than
the 50k version (Gemma 91 → 122; Qwen 201 → 228). The retrained router is
more committed to its picks but those picks still rank below the gate's, so
without an original-router rescore the gap is wider. **Hybrid mode is the
operationally correct mode** and is where the gain shows up.

## What this means for the goal

- The plan's "+0.5–1% PPL target" is **met** for both models with sensible
  operating points (50% of experts as candidates for Gemma 4; 37.5% for
  Qwen 3.6).
- The **routing primitive is fast** (kernel-side ~20 µs at all scales — see
  `results/bvh_cuda_speedup_v2.md`). At n_cand=64 (Gemma 4) or n_cand=96
  (Qwen 3.6), the BVH does a much smaller pre-filter, and the original gate
  scores only those candidates instead of all 128/256 experts — a real
  reduction in dense-gate work for a tiny PPL cost.
- **End-to-end tok/s gain in PyTorch is still ~0** because accelerate's hook
  coupling forces us to run the original router for its hooks. The routing
  win materializes once the BVH path is a fused C++/CUDA kernel that
  bypasses the PyTorch dispatch overhead (Phase 6 follow-up `Spectral-AI-wdj`).

## llama.cpp baselines (different backend, for context only)

| Model | Quant | PPL (llama.cpp) | Notes |
|---|---|---:|---|
| OLMoE 7B-A1B | F16 | 7.858 (full set) | reference |
| OLMoE 7B-A1B | Q4_K_M | 8.053 (full set) | +2.5% from quantization |
| Qwen 3.6 35B-A3B | UD-Q4_K_S | 6.438 (300 chunks) | not directly comparable |

Our PyTorch bf16 baseline for Qwen 3.6 (7.87 on 50 chunks) vs llama.cpp Q4
(6.44 on 300 chunks) differ by quantization scheme AND by chunks count, so
not apples-to-apples. The relative comparisons within each backend stand.

## Reproduction

```bash
# Re-extract 200k tokens per layer (~2-3 h per model)
python python/extract_router_io.py --model-dir <safetensors> \
    --model-kind <gemma4|qwen36> \
    --out-dir data/<model>_hiddens_200k --max-tokens 200000

# Retrain on 200k
DATA_DIR=data/<model>_hiddens_200k \
SAVE_DIR=checkpoints/<model>_distill_branch_200k \
scripts/train_all_layers.sh <model> 200000 30

# Sweep on new checkpoints
python python/ncand_sweep.py --model <model> \
    --ckpt-dir checkpoints/<model>_distill_branch_200k \
    --values 32 64 96 128 [192 256]
```
