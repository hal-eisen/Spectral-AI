#!/usr/bin/env bash
#
# Run the full e2e eval matrix for one or both models once training is complete.
#
#   scripts/run_e2e_evals.sh gemma4          # just Gemma 4
#   scripts/run_e2e_evals.sh qwen36          # just Qwen 3.6
#   scripts/run_e2e_evals.sh both            # serialized (to fit 16GB VRAM)
#
# Per model, we run 4 configs:
#   - baseline PPL
#   - baseline throughput (prefill)
#   - BVH hybrid (n_candidates=32) PPL
#   - BVH hybrid (n_candidates=32) throughput
#
# Output JSON is appended per-run to results/<model>_e2e.json so you can
# incrementally resume.
set -uo pipefail

MODEL_ARG="${1:?usage: $0 <gemma4|qwen36|both>}"
PYTHON="${PYTHON:-/home/eisen/spectralai/.venv/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

run_one() {
    local model="$1"
    local model_dir ckpt_dir script out_json
    case "$model" in
      gemma4)
        model_dir="/home/eisen/spectralai/remote_models/Google/Gemma4-26B-A4B"
        ckpt_dir="checkpoints/gemma4_distill_branch"
        script="python/gemma4_e2e_eval.py"
        out_json="results/gemma4_e2e.json"
        ;;
      qwen36)
        model_dir="/home/eisen/spectralai/remote_models/Qwen/Qwen3.6-35B-A3B"
        ckpt_dir="checkpoints/qwen36_distill_branch"
        script="python/qwen36_e2e_eval.py"
        out_json="results/qwen36_e2e.json"
        ;;
      *) echo "unknown model $model" >&2; return 2;;
    esac

    mkdir -p "$(dirname "$out_json")"
    echo "============================================================"
    echo " E2E eval — $model"
    echo " model_dir:    $model_dir"
    echo " checkpoints:  $ckpt_dir  ($(ls $ckpt_dir/*.pt 2>/dev/null | wc -l) layers trained)"
    echo " output JSON:  $out_json"
    echo "============================================================"

    # 1. Baseline PPL (no BVH)
    echo ""; echo "[1/4] Baseline PPL"
    PYTORCH_ALLOC_CONF=expandable_segments:True "$PYTHON" "$script" \
        --model-dir "$model_dir" \
        --eval-mode ppl --max-chunks 50 --ctx 512 \
        --out-json "$out_json"

    # 2. Baseline throughput
    echo ""; echo "[2/4] Baseline throughput"
    PYTORCH_ALLOC_CONF=expandable_segments:True "$PYTHON" "$script" \
        --model-dir "$model_dir" \
        --eval-mode throughput --batch-size 1 --seq-len 512 \
        --out-json "$out_json"

    # 3. BVH hybrid PPL
    echo ""; echo "[3/4] BVH hybrid PPL (n_candidates=32)"
    PYTORCH_ALLOC_CONF=expandable_segments:True "$PYTHON" "$script" \
        --model-dir "$model_dir" \
        --checkpoint-dir "$ckpt_dir" \
        --mode hybrid --n-candidates 32 \
        --eval-mode ppl --max-chunks 50 --ctx 512 \
        --out-json "$out_json"

    # 4. BVH hybrid throughput
    echo ""; echo "[4/4] BVH hybrid throughput (n_candidates=32)"
    PYTORCH_ALLOC_CONF=expandable_segments:True "$PYTHON" "$script" \
        --model-dir "$model_dir" \
        --checkpoint-dir "$ckpt_dir" \
        --mode hybrid --n-candidates 32 \
        --eval-mode throughput --batch-size 1 --seq-len 512 \
        --out-json "$out_json"

    echo ""; echo "[done] $model results → $out_json"
}

case "$MODEL_ARG" in
  gemma4|qwen36) run_one "$MODEL_ARG" ;;
  both)
    # Serialize — 16GB VRAM can't hold both 26B+35B at once.
    run_one gemma4
    run_one qwen36
    ;;
  *)
    echo "usage: $0 <gemma4|qwen36|both>" >&2
    exit 1
    ;;
esac
