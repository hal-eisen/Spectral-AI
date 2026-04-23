# Quantization–Quality Comparison (llama.cpp baseline)

Measured on RTX 4070 Ti Super (Ada sm_89, 16 GB), llama.cpp build e365e658f.
PPL measured on WikiText-2 test with `llama-perplexity -c 512 -b 512`.
Throughput measured with `llama-bench -p 512 -n 128 -r 3`.
All values **without** SpectralAI BVH routing — these are the baselines we will
compare BVH-routed numbers against.

| Model | Quant | Size (GB) | Prefill tok/s (pp512) | Generate tok/s (tg128) | PPL | ± | Config |
|---|---|---:|---:|---:|---:|---:|---|
| OLMoE (7B-A1B) | Q4_K_M | 3.92 | 14 675 ± 67 | 524 ± 3 | 8.053 | 0.053 | -ngl 99 |
| OLMoE (7B-A1B) | Q6_K   | 5.29 | 13 197 ± 118 | 438 ± 3 | 7.880 | 0.052 | -ngl 99 |
| OLMoE (7B-A1B) | Q8_0   | 6.85 | 14 936 ± 71 | 365 ± 2 | 7.860 | 0.052 | -ngl 99 |
| OLMoE (7B-A1B) | F16    | 12.89 | 6 620 ± 31 | 212 ± 0 | 7.858 | 0.052 | -ngl 99 |
| **Qwen 3.6 (35B-A3B)** | UD-Q4_K_S | 19.45 | **447 ± 4** | **37.3 ± 0.1** | **6.438** | **0.057** | -ngl 99 --n-cpu-moe 40 |
| Gemma 4 (26B-A4B) | *(pending download)* | | | | | | |

## Qwen 3.6 configuration notes

- **19.45 GB GGUF does not fit 16 GB VRAM**. Two viable configs:
  - `-ngl 32`: 26 pp128 / 35 tg32 tok/s, 8 layers on CPU (mixed)
  - `-ngl 99 --n-cpu-moe 40`: **36.3 tg32**, **150 pp128** — MoE experts on CPU, everything else on GPU. **This is the baseline we use.**
- VRAM use in baseline: 2.5 GB (model 1.9 + KV 72 MB + compute 501 MB)
- Host RAM use: 19.5 GB for expert FFN weights
- PPL measured on 300 chunks of WikiText-2 (153 k tokens) at ~385 prefill tok/s

## What's missing (and why)

- **Gemma 4 26B A4B** rows are pending until the user's GGUF download completes
  (`downloading_gemma-4-26B-A4B-it-Q4_K_M.gguf.part` on disk). A single-quant
  row will land when Q4_K_M is complete; the full F16/Q8/Q6/Q4 ladder depends
  on disk space decisions tracked in `Spectral-AI-6la`.
- **Additional Qwen 3.6 quants** (Q6_K, Q8_0, F16) are optional
  (`Spectral-AI-kkb`); a single Q4_K_S row is already informative given the
  16 GB VRAM ceiling makes anything larger impractical here.
- **BVH-routed rows** (with SpectralAI) are deferred to Phase 6 (llama.cpp
  integration) — this table captures the unmodified-llama.cpp baseline only.

## Reproduction

```bash
# OLMoE baseline (4 quants)
scripts/bench_llamacpp_quants.sh olmoe /home/eisen/spectralai/models results/olmoe_llamacpp_baseline.csv

# Qwen 3.6 (single quant, MoE-to-CPU config)
# Note: bench_llamacpp_quants.sh does NOT yet accept --n-cpu-moe; run manually:
LLAMA=/home/eisen/spectralai/llama.cpp/build/bin/llama-bench
PPL=/home/eisen/spectralai/llama.cpp/build/bin/llama-perplexity
$LLAMA -m models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf -p 512 -n 128 -ngl 99 --n-cpu-moe 40 -r 3
$PPL  -m models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf -f data/eval/wiki.test.raw -c 512 -ngl 99 --n-cpu-moe 40 --chunks 300
```
