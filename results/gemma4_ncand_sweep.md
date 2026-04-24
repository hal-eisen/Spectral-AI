# n_candidates sweep — gemma4

- Model: gemma4 (128 experts, hidden 2816)
- Checkpoints: checkpoints/gemma4_distill_branch
- Eval: WikiText-2 test, ctx=512, max_chunks=50
- Baseline PPL (no BVH): **11.8725**

| n_candidates | % of experts | PPL | Δ vs baseline |
|---:|---:|---:|---:|
| 32 | 25.0% | 12.5209 | +5.46% |
| 64 | 50.0% | 12.1058 | +1.96% |
| 96 | 75.0% | 12.0366 | +1.38% |
| 128 | 100.0% | 12.0473 | +1.47% |
| pure BVH | — | 91.5337 | +670.97% |
