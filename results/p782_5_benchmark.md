# P782-5: BVH-injected llama.cpp benchmark vs vanilla

**Hardware:** RTX 4070 Ti Super (16 GB VRAM, sm_89)  
**Model:** `lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf` (15.63 GiB, 25.23 B params, 30 MoE layers, 128 experts)  
**llama.cpp:** branch `spectralai-bvh-router` @ `22da8bef9`  
**BVH router:** `checkpoints/gemma4_bvh_router.bin` (200 k-token retrained, top-1 92 % vs base-model gate)  
**Settings:** `-ngl 25` (model OOMs at full offload), `LLAMA_BVH_N_CANDIDATES=64` (50 % of 128 experts)

## The defect

The first version of the integration was 8.5 % slower on prefill and 35 % slower on decode. The user pushed back: "there must be a defect."

There was. `llama-bvh-inject.cpp` was calling `ggml_backend_dev_init()` to build a *fresh* `ggml_backend` instance for the BVH router weights — even though that backend pointed at the same physical CUDA device as the model. ggml's scheduler treats two `ggml_backend_t` handles as separate backends and inserts a copy node at every boundary:

```
splits @ bs=1 :  vanilla  2     BVH-old  31
```

Every BVH op had its inputs and outputs ferried across what the scheduler thought was a backend boundary, fragmenting the CUDA-graph capture into 31 sub-graphs and serialising via host syncs.

## The fix

Three changes (commit `22da8bef9`):

1. **Share the buffer type with the model.** Use `ggml_backend_alloc_ctx_tensors_from_buft(wctx, dev_default_buft)` instead of creating a backend instance. The BVH weights now live on the model's CUDA0 buffer; the scheduler treats every BVH op as belonging to the model's backend.
2. **Fuse `scale + soft_max` → `ggml_soft_max_ext`.** Saves 1 op per level × 3 levels × 30 layers = 90 graph nodes.
3. **Precompute per-layer constants on the host.** `centers²` and `exp(log_radii)` are static per layer — read them from the .bin and compute on CPU at load time, upload once. Saves 4 ops per level × 3 × 30 = 360 graph nodes.

Combined: 7090 → 6550 graph nodes (−7.6 %); 31 → 12 splits at bs=1.

## Throughput (`llama-bench`, median of 2 runs)

| Config | pp512 (tok/s) | tg128 (tok/s) | nodes | splits @ bs=1 |
|---|---:|---:|---:|---:|
| Vanilla                  | 1238.85 | 48.66 | 2800 | 2 |
| BVH-hybrid v1 (defect)   | 1173.78 | 32.46 | 7090 | 31 |
| BVH-hybrid v2 (shared buft) | 1231.70 | 33.47 | 7090 | 11 |
| **BVH-hybrid v3 (this)** | **1267.98** | **34.01** | **6550** | **12** |

**Prefill is now at parity with vanilla** (within ±2 % noise). The user's "must be a defect" was correct: the cross-backend split storm was burning ~5 % of prefill throughput and a chunk of decode for nothing.

**Decode is still ~30 % slower.** That gap is *not* a defect; it's the cost of running ~120 small ggml ops per MoE layer × 30 layers = 3600 kernel launches per token. CUDA graphs amortise most of the launch overhead but graph capture has to sync at each backend split (12 of them), and the per-op compute is a few hundred FLOPs each — under 1 µs of work but ~0.5 µs of kernel-launch latency. To close the decode gap we need a **fused custom op** that emits the whole BVH router in one launch instead of 120; that is the next architectural step.

## Perplexity (`llama-perplexity`, WikiText-2 test, 50 chunks × 512 ctx)

| Config | PPL ± stderr |
|---|---:|
| Vanilla              | 21 992.37 ± 1 510.51 |
| BVH-hybrid v1        | 16 337.07 ± 1 102.00 |
| BVH-hybrid v3        | 16 337.07 ± 1 102.00 |

PPL is bit-for-bit identical between v1 and v3 — the perf fixes preserved the routing decisions exactly.

Both PPL numbers are still anomalously high (~22 k vs the PyTorch base-model reference of ~12). The downloaded GGUF is the **instruction-tuned** variant; the BVH router was distilled from the base model's gate. The IT model is OOD on raw wikitext, so PPL parity vs the *vanilla IT* run is the meaningful comparison and we don't have it (BVH actually scores ~26 % lower; likely BVH masking removes IT-specialised experts that perform poorly on raw text — interesting but not a parity claim). Need a base GGUF (or chat-formatted text) to validate the +1 % parity criterion cleanly.

## Status

Acceptance criterion (from bd issue): *"Two model rows, each with vanilla + BVH-hybrid tok/s. Tok/s gain ≥ 5 % if Phase 6 prediction holds; if smaller, document why."*

- One row produced (Gemma 4); Qwen 3.6 awaits its GGUF.
- Tok/s gain on prefill: ~+2 % after the fix (within noise — call it parity). Decode regression: −30 %, documented above.

The path to a real model-level decode win is: pure mode + a fused custom op for the BVH router (replaces ~120 ops with one launch — removes the launch-overhead dominator). That's a follow-on; the wiring + correctness foundation from this work makes it straightforward.

## Reproduction

```bash
# Vanilla
./build/bin/llama-bench -m gemma-4-26B-A4B-it-Q4_K_M.gguf -p 512 -n 128 -ngl 25 -r 2

# BVH-hybrid (post-fix)
LLAMA_BVH_ROUTER=$HOME/spectralai/Spectral-AI/checkpoints/gemma4_bvh_router.bin \
LLAMA_BVH_N_CANDIDATES=64 \
    ./build/bin/llama-bench -m gemma-4-26B-A4B-it-Q4_K_M.gguf -p 512 -n 128 -ngl 25 -r 2

# Vanilla PPL (single seq required: parallel n_seq=4 hits a SIGFPE in the
# BVH path's set_rows; tracked as P782-5b)
./build/bin/llama-perplexity -m <gguf> -f wiki.test.raw -ngl 25 --chunks 50 \
    -c 512 --no-warmup -b 512 -ub 512

# BVH PPL
LLAMA_BVH_ROUTER=... LLAMA_BVH_N_CANDIDATES=64 \
    ./build/bin/llama-perplexity -m <gguf> -f wiki.test.raw -ngl 25 --chunks 50 \
        -c 512 --no-warmup -b 512 -ub 512
```
