# n_candidates sweep — gemma4

- Model: gemma4 (128 experts, hidden 2816)
- Checkpoints: checkpoints/gemma4_distill_branch_200k
- Eval: WikiText-2 test, ctx=512, max_chunks=50
- Baseline PPL (no BVH): **11.8725**

| n_candidates | % of experts | PPL | Δ vs baseline |
|---:|---:|---:|---:|
| 32 | 25.0% | 12.2109 | +2.85% |
| 64 | 50.0% | 11.9380 | +0.55% |
| 96 | 75.0% | 11.9664 | +0.79% |
| 128 | 100.0% | 11.9837 | +0.94% |
| pure BVH | — | 121.9922 | +927.52% |
