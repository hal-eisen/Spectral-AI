#!/usr/bin/env bash
#
# Train a BranchSpecificBVHRouter for every MoE layer of a model.
#
# Works for both Gemma 4 (30 layers, 128 experts, 2816 dim) and Qwen 3.6
# (40 layers, 256 experts, 2048 dim) by parameter. Expects extracted
# router-I/O .pt files in <data_dir>/router_io_layer{00..NN}.pt — the
# format produced by python/extract_router_io.py.
#
# Usage:
#   scripts/train_all_layers.sh gemma4
#   scripts/train_all_layers.sh qwen36
#
# Resumes: skips layers whose checkpoint already exists. Delete
# <save_dir>/bvh_router_L{NN}_best.pt to force retrain.

set -uo pipefail

MODEL="${1:?usage: $0 <gemma4|qwen36> [n_train] [epochs]}"
N_TRAIN="${2:-50000}"
EPOCHS="${3:-30}"

case "$MODEL" in
  gemma4)
    DATA_DIR="data/gemma4_hiddens"
    SAVE_DIR="checkpoints/gemma4_distill_branch"
    N_EXPERTS=128
    EMBED_DIM=2816
    N_LAYERS=30
    ;;
  qwen36)
    DATA_DIR="data/qwen36_hiddens"
    SAVE_DIR="checkpoints/qwen36_distill_branch"
    N_EXPERTS=256
    EMBED_DIM=2048
    N_LAYERS=40
    ;;
  *)
    echo "ERROR: unknown model '$MODEL' (use gemma4 or qwen36)" >&2
    exit 1
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-/home/eisen/spectralai/.venv/bin/python}"
SCRIPT="python/olmoe_bvh_distill.py"

mkdir -p "$SAVE_DIR"
LOGFILE="$SAVE_DIR/training_log.txt"
: > "$LOGFILE"

echo "============================================================" | tee -a "$LOGFILE"
echo " BVH Router Training — $MODEL" | tee -a "$LOGFILE"
echo " Layers: $N_LAYERS   Experts: $N_EXPERTS   Hidden: $EMBED_DIM" | tee -a "$LOGFILE"
echo " n_train: $N_TRAIN   epochs: $EPOCHS" | tee -a "$LOGFILE"
echo " Data:  $DATA_DIR" | tee -a "$LOGFILE"
echo " Save:  $SAVE_DIR" | tee -a "$LOGFILE"
echo " Start: $(date -Iseconds)" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

OK=0
SKIP=0
FAIL=0
for ((L=0; L < N_LAYERS; L++)); do
    LPAD=$(printf "%02d" "$L")
    DATA="$DATA_DIR/router_io_layer${LPAD}.pt"
    CKPT="$SAVE_DIR/bvh_router_L${L}_best.pt"

    echo "" | tee -a "$LOGFILE"
    echo "---- Layer $L / $((N_LAYERS-1)) ----" | tee -a "$LOGFILE"

    if [[ ! -f "$DATA" ]]; then
        echo "[SKIP] no data at $DATA" | tee -a "$LOGFILE"
        SKIP=$((SKIP+1))
        continue
    fi
    if [[ -f "$CKPT" ]]; then
        echo "[SKIP] checkpoint already exists: $CKPT (delete to retrain)" | tee -a "$LOGFILE"
        SKIP=$((SKIP+1))
        continue
    fi

    echo "[TRAIN] $DATA → $CKPT" | tee -a "$LOGFILE"
    T0=$(date +%s)
    "$PYTHON" "$SCRIPT" \
        --layer "$L" \
        --real-data "$DATA" \
        --no-upcycle \
        --spectral \
        \
        --n-experts "$N_EXPERTS" \
        --embed-dim "$EMBED_DIM" \
        --epochs "$EPOCHS" \
        --batch-size 2048 \
        --n-train "$N_TRAIN" \
        --save-dir "$SAVE_DIR" \
        --device cuda \
        >> "$LOGFILE" 2>&1
    RC=$?
    DT=$(( $(date +%s) - T0 ))

    if [[ $RC -eq 0 ]]; then
        echo "[OK]   layer $L completed in ${DT}s" | tee -a "$LOGFILE"
        OK=$((OK+1))
    else
        echo "[FAIL] layer $L (rc=$RC, ${DT}s)" | tee -a "$LOGFILE"
        FAIL=$((FAIL+1))
    fi
done

echo "" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
echo " Summary: OK=$OK  SKIP=$SKIP  FAIL=$FAIL" | tee -a "$LOGFILE"
echo " Done: $(date -Iseconds)" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

exit $([[ $FAIL -eq 0 ]] && echo 0 || echo 2)
