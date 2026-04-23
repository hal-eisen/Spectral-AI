# Qwen 3.6 35B A3B — extraction strategy decision

**Decision:** download HF safetensors, load with stock transformers 5.5.0 — no GGUF dequant, no `trust_remote_code`.

## Why

`transformers 5.5.0` ships **native `qwen3_5_moe` support**:

- Module: `transformers.models.qwen3_5_moe.modeling_qwen3_5_moe`
- Classes: `Qwen3_5MoeConfig`, `Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MoeTextConfig`, `Qwen3_5MoeTextModel`, `Qwen3_5MoeVisionModel`
- AutoConfig already loads `Qwen/Qwen3.6-35B-A3B` cleanly:
  - `model_type: qwen3_5_moe`
  - `text_config` is `Qwen3_5MoeTextConfig`

So the dequant-from-GGUF approach (added cost: write `qwen36_extract.py` against `llama-cpp-python`) is unnecessary.

## What we'll do

1. Mount a remote host (via sshfs) carrying `Qwen/Qwen3.6-35B-A3B` safetensors (26 shards, ~52GB). **No local download**; disk issue Spectral-AI-6la resolved by this architecture choice.
2. Load with `Qwen3_5MoeForCausalLM` (text-only — multimodal vision is out of scope per user decision). Stock `transformers 5.5.0`, no `trust_remote_code`.
3. Use the existing extraction pattern from `python/gemma4_extract.py` adapted for the qwen3_5_moe block tree.
4. sshfs read bandwidth will affect one-time extraction wallclock — budget for it in P3-3.

## Architecture facts learned (relevant to BVH router config)

From the GGUF metadata + HF config:

| Field | Value | Why it matters |
|---|---|---|
| `num_hidden_layers` | 40 | Train BVH router for each |
| `hidden_size` | 2048 | `--embed-dim 2048` for `BranchSpecificBVHRouter` |
| `num_experts` | 256 | Tree factorization: 4×8×8 or 8×4×8 |
| `num_experts_per_tok` | 8 | `--top-k 8` like OLMoE |
| `moe_intermediate_size` | 512 | per-expert FFN size |
| `shared_expert_intermediate_size` | 512 | **shared expert exists** — leave its gate (`ffn_gate_inp_shexp`) untouched, only swap the routed-expert gate |
| `layer_types` | mix of `linear_attention` (hybrid Mamba) and `full_attention` every 4th layer | Both layer types still have MoE FFN; gate replacement is layer-type-agnostic |
| `vocab_size` | 248320 | larger than OLMoE — affects embedding RAM |
| `max_position_embeddings` | 262144 | Fine; we'll eval at ctx=512 like OLMoE |
| MTP | `mtp_num_hidden_layers: 1` | Multi-token-prediction head; ignore for routing eval |

## Implications for P3-2 (write `python/qwen36_extract.py`)

- Loop over 40 MoE blocks. Each has `mlp.gate` (256-way routed gate) and a separate `mlp.shared_expert` path.
- We hook `block.mlp.gate` to capture `(hidden_state, top-k expert IDs, top-k probs)`.
- The hybrid Mamba layers are NOT a problem for extraction; they still expose `hidden_states` post-block.
- Extraction needs ~52GB on disk for safetensors plus ~26GB VRAM peak (4-bit quant load via bitsandbytes brings it to ~9GB; safer with `device_map="auto"`).

## Implications for `BranchSpecificBVHRouter` config

```
BranchSpecificBVHRouter(
    cfg=RouterConfig(
        embed_dim=2048,
        n_level1=4,
        n_level2=8,
        n_level3=8,   # 4*8*8 = 256 experts
        spectral_dim=64,
        ...
    )
)
```

## Out of scope (text-only project decision)

- `vision_config` (qwen3_5_moe vision encoder)
- `image_token_id`, `video_token_id`
- The mmproj GGUF Unsloth shipped
