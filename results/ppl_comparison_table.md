# PPL — Baseline vs SpectralAI BVH routing

Measured on **RTX 4070 Ti Super (Ada sm_89, 16 GB)**, native Linux, PyTorch
2.10 bf16 with accelerate auto-spill. PPL computed on WikiText-2 test split
with `ctx=512`, 50 chunks (25 600 tokens per run), CE loss on shifted-by-1
targets. Same chunks and tokenizer for all runs of a given model.

The BVH routing replaces each MoE layer's expert-selection step with a
trained `EnhancedBVHRouter` (3-level BVH, 3³=27D effective routing) that
was distilled from the original gate on 50k WikiText-2 tokens per layer
over 30 epochs. In **hybrid** mode the BVH narrows to N candidates and
the original router scores them exactly (preserving per_expert_scale
calibration); in **pure** mode BVH selects top-k directly (not eval'd
here — hybrid is the primary target).

## Results

| Model | Mode | n_cand | PPL | Δ vs baseline |
|---|---|---:|---:|---:|
| Gemma 4 26B A4B | baseline | — | **11.873** | — |
| Gemma 4 26B A4B | BVH hybrid | 32 of 128 (25%) | **12.587** | +6.1% |
| Qwen 3.6 35B A3B | baseline | — | **7.873** | — |
| Qwen 3.6 35B A3B | BVH hybrid | 32 of 256 (12.5%) | **8.698** | +10.5% |

## Why Qwen's degradation is larger

Three factors compound:

1. **Tighter candidate ratio**: n_candidates=32 is 25% of Gemma 4's 128
   experts but only 12.5% of Qwen's 256. Whenever the true top-8 expert for
   a token isn't in the BVH's top-32, hybrid mode cannot recover.
2. **Lower router accuracy**: Qwen 3.6 router mean top-8 overlap is 59.3%
   vs Gemma 4's 73.4% (see `results/{gemma4,qwen36}_router_acc.md`).
3. **Fewer training tokens per expert**: 50 000 tokens / 256 experts =
   ~195 tokens per expert for Qwen vs ~390 per expert for Gemma 4.

## Follow-ups that should close the gap

Open bd issue `Spectral-AI-xxx` tracks each.

- **Retrain with more data** (200k tokens/layer matches the OLMoE paper
  recipe that achieved top-8 ~95%+).
- **Increase n_candidates** at inference: 64 for Gemma 4 (50%), 96 for
  Qwen 3.6 (37.5%). Cost per layer is a larger topk but same hidden-state
  compute.
- **Pure-BVH mode on a better-trained router**: skips the original proj
  entirely (real speedup signal), accepts some quality loss directly.

## Comparison to llama.cpp baselines

For reference, llama.cpp baseline PPL on the same WikiText-2 test (full
set, ctx=512) is in `results/olmoe_llamacpp_baseline.csv` and
`results/qwen36_llamacpp.csv`:

| Model | Quant | PPL | Δ vs F16 |
|---|---|---:|---:|
| OLMoE 7B-A1B | F16 | 7.858 | baseline |
| OLMoE 7B-A1B | Q4_K_M | 8.053 | +2.5% |
| Qwen 3.6 35B-A3B | UD-Q4_K_S | 6.438 (300 chunks) | n/a |

Our PyTorch bf16 Qwen 3.6 baseline (7.87 PPL on 50 chunks) vs llama.cpp Q4
Qwen 3.6 (6.44 PPL on 300 chunks) is not a fair comparison (different
quant and chunk count); the 50-vs-300-chunk gap typically accounts for
~0.5 PPL and Q4→bf16 typically widens PPL by several percent.

## Reproduction

```bash
scripts/run_e2e_evals.sh gemma4
scripts/run_e2e_evals.sh qwen36
# Results in results/{gemma4,qwen36}_e2e.json
```
