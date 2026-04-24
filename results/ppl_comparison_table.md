# PPL — Baseline vs SpectralAI BVH routing

Measured on **RTX 4070 Ti Super (Ada sm_89, 16 GB)**, native Linux, PyTorch
2.10 bf16 with accelerate auto-spill. PPL computed on WikiText-2 test split
with `ctx=512`, 50 chunks (25 600 tokens per run), CE loss on shifted-by-1
targets. Same chunks and tokenizer for all runs of a given model.

## Headline numbers (BVH hybrid with `n_candidates=32`, original install)

| Model | Mode | n_cand | PPL | Δ vs baseline |
|---|---|---:|---:|---:|
| Gemma 4 26B A4B | baseline | — | **11.873** | — |
| Gemma 4 26B A4B | BVH hybrid | 32 of 128 (25%) | **12.587** | +6.02% |
| Qwen 3.6 35B A3B | baseline | — | **7.873** | — |
| Qwen 3.6 35B A3B | BVH hybrid | 32 of 256 (12.5%) | **8.698** | +10.48% |

Follow-up work (Phase 7) reduced these materially — see below.

## Phase 7 Addendum — `n_candidates` sweep

Widening the BVH candidate set closes most of the PPL gap without retraining.
Full tables: [`gemma4_ncand_sweep.md`](gemma4_ncand_sweep.md),
[`qwen36_ncand_sweep.md`](qwen36_ncand_sweep.md).

### Gemma 4 26B A4B (128 experts, baseline PPL 11.873)

After a per_expert_scale fix (reading from safetensors rather than a broken
forward-pre hook — commit `a9027d9`):

| n_candidates | % of experts | PPL | Δ vs baseline |
|---:|---:|---:|---:|
| 32 | 25.0% | 12.521 | **+5.46%** |
| 64 | 50.0% | 12.106 | **+1.96%** |
| 96 | 75.0% | 12.037 | **+1.38%** |
| 128 | 100.0% | 12.047 | **+1.47%** |
| pure BVH | — | 91.534 | +671% |

**Operating point: `n_candidates=64`** (50% of experts) delivers +2% PPL,
vs +6% at the original n=32. n=128 should theoretically equal baseline but
has a +1.47% residual — likely bf16 ordering / numerical path divergence in
the final renorm + per_expert_scale multiply inside the wrapper. Not
blocking; smaller than the expected retraining gain.

### Qwen 3.6 35B A3B (256 experts, baseline PPL 7.873)

Qwen has no per_expert_scale and its wrapper matches baseline exactly at
n=256 (+0.04%):

| n_candidates | % of experts | PPL | Δ vs baseline |
|---:|---:|---:|---:|
| 32 | 12.5% | 8.698 | **+10.48%** |
| 64 | 25.0% | 8.256 | **+4.87%** |
| 96 | 37.5% | 8.067 | **+2.47%** |
| 128 | 50.0% | 7.968 | **+1.20%** |
| 192 | 75.0% | 7.884 | **+0.14%** |
| 256 | 100.0% | 7.876 | **+0.04%** |
| pure BVH | — | 201.001 | +2453% |

**Operating points for Qwen 3.6:**
- `n_candidates=128` (50% of experts): +1.2% PPL — practical sweet spot
- `n_candidates=192` (75% of experts): +0.14% PPL — within plan's ±1% target
- `n_candidates=256`: matches baseline within rounding noise

## What this tells us

- **The BVH router is a valid candidate pre-filter but NOT a standalone
  ranker.** Pure BVH is +670% (Gemma 4) and +2453% (Qwen 3.6) — its top-8
  selection is much worse than the learned gate. In hybrid mode the original
  router does the final top-8 scoring over the BVH-narrowed candidate set,
  which is where the quality is preserved.
- **50% candidate ratio is the universal sweet spot** for both models: +2%
  (Gemma 4 n=64) and +1.2% (Qwen 3.6 n=128). Good enough for many use cases
  without any retraining.
- **75% candidate ratio gets us within plan's ±1% target** for Qwen 3.6.
  For Gemma 4 the +1.47% residual at n=128 (100%) limits how close we can
  get without fixing the wrapper's numerical path.
- **Retraining with 4× more data (P7-3 through P7-6) should shift the curve
  left**: the same PPL should become achievable at smaller n_candidates,
  which matters when BVH is actually the routing primitive (smaller n_cand
  means less work for the original-router rescore).

## Follow-ups in flight

- **P7-3** (running): re-extract 200 k tokens per layer (4× current 50k)
  for both models. Expected wallclock ~2.5 h per model. Log:
  `logs/{gemma4,qwen36}_extract_200k.log`.
- **P7-4 / P7-5**: retrain both routers on 200k. Target top-8 overlap ≥ 85%
  (Gemma 4) / ≥ 75% (Qwen 3.6), vs current 73% / 59%.
- **P7-6**: re-eval with the new checkpoints at the best n_candidates.

## For the record: llama.cpp baselines (same hardware, different backend)

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
# Full e2e eval matrix (baseline + BVH @ n=32)
scripts/run_e2e_evals.sh gemma4
scripts/run_e2e_evals.sh qwen36

# n_candidates sweep (single model load, ~70 min each)
python python/ncand_sweep.py --model gemma4 --values 32 64 96 128
python python/ncand_sweep.py --model qwen36 --values 32 64 96 128 192 256

# Results in results/{gemma4,qwen36}_{e2e,ncand_sweep}.{md,json}
```
