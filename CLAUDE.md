# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Build & Test

### Python (primary development)

```bash
pip install -e ".[dev,eval]"              # Install with test + eval deps

# Run all tests (209 tests)
python -m pytest tests/ -v --tb=short

# Run a specific test file
pytest tests/test_bvh_router.py -v        # Core routing (79 tests)
pytest tests/test_enhanced_bvh_router.py  # Enhanced routing (47 tests)
pytest tests/test_bvh_gate_wrapper.py     # Gate wrapper (48 tests)
pytest tests/test_polysemy_benchmark.py   # Polysemy (21 tests)
pytest tests/test_optix_integration.py    # OptiX integration (14 tests)

# Run a single test
pytest tests/test_bvh_router.py::TestBVHRouter::test_forward -v
```

### C++/CUDA (OptiX RT Core pipeline)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)

# With OptiX (optional, for RT Core pipeline)
cmake .. -DOptiX_INSTALL_DIR=/path/to/optix -DCMAKE_BUILD_TYPE=Release
```

### Training & Evaluation

```bash
# Extract hidden states from OLMoE (prerequisite for training)
python python/extract_real_hiddens.py --model-dir /path/to/olmoe-1b-7b

# Train single layer
python python/olmoe_bvh_distill.py --model-dir /path/to/olmoe-1b-7b \
    --layer 8 --epochs 100 --spectral --real-data data/real_hiddens_layer8.pt

# End-to-end PPL evaluation (single layer)
python python/olmoe_e2e_eval.py --model-dir /path/to/olmoe-1b-7b \
    --router-checkpoint checkpoints/olmoe_distill/bvh_router_best.pt \
    --layer 8 --max-tokens 50000

# Multi-layer hybrid evaluation
python python/olmoe_e2e_eval.py --model-dir /path/to/olmoe-1b-7b \
    --n-candidates 48 --hybrid --max-tokens 20000 \
    --multi-layer "0:checkpoints/olmoe_best/bvh_router_L0_best.pt,..."

# Downstream task eval
python python/eval_hellaswag.py --model-dir /path/to/olmoe-1b-7b --max-samples 2000
```

## Architecture Overview

**SpectralAI** replaces O(N) dense routing gates in Mixture-of-Experts models with O(log N) BVH (Bounding Volume Hierarchy) traversal on NVIDIA RT Cores — the same silicon used for ray tracing in games.

### Core pipeline

```
Tokens → Embedding → 3D Projection (PCA, 2048→3D) → BVH Router (4 levels)
→ Top-k Expert Selection → Expert FFN (frozen OLMoE) → Output
```

### Three key innovations

1. **RT Core Routing** (`python/bvh_router.py`, `cuda/`) — Expert centroids organized as AABB bounding boxes in a BVH tree. Ray traversal finds top-k experts in O(log N) instead of O(N) dot products. 48-89x speedup, 731x less VRAM.

2. **Inception Engine** (`python/inception_attention.py`, `cuda/inception_kernels.cu`) — 4 nested IAS levels: 4 Domains → 16 Subdomains → 64 Experts. Hierarchical routing with Gumbel-Softmax (training) / hard argmax (inference).

3. **Spectral Routing** (`python/spectral_techniques.py`, `python/eval_polysemy.py`) — Tokens carry a "wavelength" context vector. At domain boundaries, Snell's law refracts compatible contexts through while incompatible ones trigger Total Internal Reflection, resolving polysemy (98.4% accuracy) without duplicating parameters.

### Key directories

- **`python/`** — All PyTorch code: router, distillation, evaluation, benchmarks (~50 files)
- **`cuda/`** — CUDA/OptiX kernels: ray generation, closest-hit shaders, Tensor Core calibration
- **`include/`** / **`src/`** — C++ headers and implementations for BVH construction
- **`tests/`** — pytest suite (209 tests) + CUDA benchmarks
- **`checkpoints/olmoe_best/`** — Per-layer trained routers: `bvh_router_L{0..15}_best.pt`
- **`scripts/`** — Automation: calibration, multi-layer eval, paper table generation
- **`zenodo/`** — Preprints and publication materials

### Key results (OLMoE-1B-7B, 64 experts, 16 MoE layers)

- **Routing latency**: 927 µs → 19.1 µs (48x speedup via RT Cores)
- **Router VRAM**: 2,944 MB → 4.03 MB (731x reduction)
- **Perplexity**: 7.00 → 7.00 (zero degradation with pre-filter 48 candidates)
- **Per-layer accuracy**: 95.4%–97.6% (mean 96.6%), all layers > 95%

### Performance modes

| Mode | PPL | Notes |
|------|-----|-------|
| Pre-filter 48 candidates | 7.00 (+0.0%) | Recommended balanced mode |
| Pure 3 layers | 7.33 (+2.5%) | No gate, fastest inference |
| Hybrid 16 layers | 7.30 (+2.1%) | Original gate as fallback |

## Conventions & Patterns

- **Differentiable ↔ non-differentiable bridge**: The PyTorch router (`bvh_router.py`) is differentiable; OptiX RT Core routing is not. `bvh_router_bridge.py` bridges them via StraightThroughEstimator (STE).
- **Per-layer checkpoints**: Each of 16 MoE layers has its own trained router. Layer accuracy varies; always load the correct per-layer checkpoint.
- **CUDA hot paths**: No `std::vector`, use flat arrays or `thrust::device_vector`. All GPU memory must have corresponding free.
- **Target hardware**: RTX 5070 Ti (Blackwell sm_120) primary, RTX 4090 (Ada sm_89) also supported. RT Cores required.
- **WSL2 vs Windows**: Python evaluation runs in WSL2. OptiX/RT Core testing requires Windows native CUDA.
- **Decision context**: `LEARNINGS.md` (2,640 lines) documents all design decisions, failures, and alternatives considered — consult it before changing architecture.
- **Dependencies**: Core = torch + numpy + scikit-learn. Eval extras = transformers + datasets + accelerate + safetensors (see `pyproject.toml`).
