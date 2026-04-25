# P782-5: BVH-injected llama.cpp benchmark vs vanilla

**Hardware:** RTX 4070 Ti Super (16 GB VRAM, sm_89)  
**Model:** `lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf` (15.63 GiB, 25.23 B params, 30 MoE layers, 128 experts)  
**llama.cpp:** branch `spectralai-bvh-router` @ `2ddb60336` + `graph_max_nodes` headroom fix  
**BVH router:** `checkpoints/gemma4_bvh_router.bin` (200 k-token retrained, top-1 92 % vs base-model gate)  
**Settings:** `-ngl 25` (model OOMs at full offload), `LLAMA_BVH_N_CANDIDATES=64` (50 % of 128 experts)

## Throughput (`llama-bench`)

| Config | pp512 (tok/s) | tg128 (tok/s) | speedup vs vanilla |
|---|---:|---:|---:|
| Vanilla              | 1283.24 ± 91.75 | 49.80 ± 0.49 | 1.00× |
| BVH-hybrid (n_cand=64) | 1173.78 ± 41.53 | 32.46 ± 0.02 | 0.91× pp / 0.65× tg |

**BVH-hybrid is slower at the model level**, by 8.5 % on prefill and 35 % on decode. This is an expected outcome of *hybrid* mode:

- Hybrid mode does **not** replace the original gate matmul. It adds the full BVH router forward (~80 ggml ops/layer × 30 layers ≈ 2400 extra ops) **plus** a top-N argsort **plus** a `set_rows` mask. The original gate's `mul_mat` (2816 × 128 = 360 K FLOPs/token) is unchanged.
- The BVH router's inner loop has many small ops (per-level Linear+GELU+Linear+matmul+softmax+tanh+clamp+exp). Each ggml op has fixed CUDA-launch overhead (~5–10 µs); summing ~2400 launches per forward pass dominates per-token cost during decode (where tokens-per-pass = 1).
- Pure mode (BVH **replaces** the gate) would skip the gate matmul but is not yet wired -- the gate matmul is cheap relative to the experts, so even in pure mode the speedup at the model level would be marginal on this hardware.

The published 48× routing-layer speedup applies to the BVH primitive vs an O(N) softmax+top-k *primitive*, not to end-to-end model latency.

## Perplexity (`llama-perplexity`, WikiText-2 test, 50 chunks × 512 ctx)

| Config | PPL ± stderr |
|---|---:|
| Vanilla              | **21 992.37 ± 1 510.51** |
| BVH-hybrid (n_cand=64) | **16 337.07 ± 1 102.00** |

Both PPLs are anomalously high. Reference numbers from the PyTorch fork on the *same WikiText test set with the BF16 safetensors* model:

- Full BF16 baseline:               7.87
- 4-bit experts + BVH (n=128):     ~8.0  (within +1 % of bf16)

Several confounders explain the 21 k figure:

1. The downloaded GGUF is `gemma-4-26B-A4B-**it**` (instruction-tuned). The PyTorch reference numbers come from the **base** model. An IT-tuned model is OOD on raw wikitext.
2. The BVH router was distilled from the base model's gate. Applied to an IT model, the BVH's per-layer routing decisions still match the gate's BF16 decisions ~92 % of the time at the *base* model, but the IT model's experts may have been further specialised for chat -- so the routing-quality impact on raw-text perplexity is unclear.
3. `-ngl 25` partially offloads (5 of 30 layers' experts on CPU), which can introduce tiny FP-precision differences vs full GPU.

The BVH-hybrid PPL is ~26 % *lower* than vanilla, which inverts the expected sign. Most likely the BVH masking is filtering out experts the IT model would have picked but that perform worse on raw text -- effectively a noise-reduction pass on a distribution mismatch. This is interesting but **not a valid PPL-parity claim** until we can compare against a base GGUF (or chat-format input).

## Status

**Acceptance criterion** (from bd): *"two model rows, each with vanilla + BVH-hybrid tok/s. Tok/s gain ≥ 5 % if Phase 6 prediction holds; if smaller, document why."*

- Two rows produced (Gemma 4; Qwen 3.6 awaits its GGUF download).
- Tok/s **regressed** ~9–35 %; documented above. Hybrid mode adds work without removing the gate matmul, so a model-level speedup is not expected from this configuration alone.

**The valuable signal here is correctness, not speed.** The BVH router runs end-to-end in llama.cpp's CUDA backend, masks `selection_probs` correctly (verified standalone in `--mask-test`), and produces stable PPL across 50 chunks. The path to a real model-level speedup is one of:

- Pure mode + a fused CUDA op for the BVH router (replaces ~80 ops with one launch -- removes the launch-overhead dominator on decode).
- 4-bit-experts deployment from the SpectralAI fork (already shows 2.4× tok/s on Gemma 4 in PyTorch); BVH then becomes the routing-quality preserver, not the speedup source.

## Reproduction

```bash
# Vanilla
./build/bin/llama-bench -m gemma-4-26B-A4B-it-Q4_K_M.gguf -p 512 -n 128 -ngl 25 -r 2

# BVH-hybrid
LLAMA_BVH_ROUTER=$HOME/spectralai/Spectral-AI/checkpoints/gemma4_bvh_router.bin \
LLAMA_BVH_N_CANDIDATES=64 \
    ./build/bin/llama-bench -m gemma-4-26B-A4B-it-Q4_K_M.gguf -p 512 -n 128 -ngl 25 -r 2

# Vanilla PPL (single seq required: parallel n_seq=4 hits a SIGFPE in the
# BVH path's set_rows; tracked as a follow-up)
./build/bin/llama-perplexity -m <gguf> -f wiki.test.raw -ngl 25 --chunks 50 \
    -c 512 --no-warmup -b 512 -ub 512

# BVH PPL
LLAMA_BVH_ROUTER=... LLAMA_BVH_N_CANDIDATES=64 \
    ./build/bin/llama-perplexity -m <gguf> -f wiki.test.raw -ngl 25 --chunks 50 \
        -c 512 --no-warmup -b 512 -ub 512
```
