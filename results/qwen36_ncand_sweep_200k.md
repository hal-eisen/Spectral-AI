# n_candidates sweep — qwen36

- Model: qwen36 (256 experts, hidden 2048)
- Checkpoints: checkpoints/qwen36_distill_branch_200k
- Eval: WikiText-2 test, ctx=512, max_chunks=50
- Baseline PPL (no BVH): **7.8729**

| n_candidates | % of experts | PPL | Δ vs baseline |
|---:|---:|---:|---:|
| 32 | 12.5% | 8.1251 | +3.20% |
| 64 | 25.0% | 7.9552 | +1.05% |
| 96 | 37.5% | 7.8967 | +0.30% |
| 128 | 50.0% | 7.8744 | +0.02% |
| 192 | 75.0% | 7.8795 | +0.08% |
| 256 | 100.0% | 7.8749 | +0.03% |
| pure BVH | — | 228.0508 | +2796.65% |
