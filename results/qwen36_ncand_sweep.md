# n_candidates sweep — qwen36

- Model: qwen36 (256 experts, hidden 2048)
- Checkpoints: checkpoints/qwen36_distill_branch
- Eval: WikiText-2 test, ctx=512, max_chunks=50
- Baseline PPL (no BVH): **7.8729**

| n_candidates | % of experts | PPL | Δ vs baseline |
|---:|---:|---:|---:|
| 32 | 12.5% | 8.6978 | +10.48% |
| 64 | 25.0% | 8.2560 | +4.87% |
| 96 | 37.5% | 8.0674 | +2.47% |
| 128 | 50.0% | 7.9675 | +1.20% |
| 192 | 75.0% | 7.8842 | +0.14% |
| 256 | 100.0% | 7.8764 | +0.04% |
| pure BVH | — | 201.0008 | +2453.07% |
