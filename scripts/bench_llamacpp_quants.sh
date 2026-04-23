#!/usr/bin/env bash
#
# Run llama-bench (prompt + generation tok/s) and llama-perplexity on a set of
# GGUFs sharing the same base model, and emit a single CSV row per quant.
#
# Used for:
#   - Phase 2: Gemma 4 quantization-quality row    (4 GGUFs)
#   - Phase 4: Qwen 3.6 quantization-quality row   (1+ GGUFs)
#   - OLMoE baseline (reproducible)
#
# Usage:
#   scripts/bench_llamacpp_quants.sh <model_name> <gguf_dir> <out_csv>
#
# The script discovers GGUFs matching the pattern "<model_name>-*.gguf" inside
# <gguf_dir>. For each, it reads size, runs llama-bench -p 512 -n 128 -ngl 99,
# and runs llama-perplexity on the project's WikiText-2 test file.
#
# It does NOT download anything. Point it at already-available GGUFs.

# Intentionally NOT using `set -e`: grep exits 1 on no-match, which we don't
# want to abort the whole script. We handle errors per-command.
set -uo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <model_name> <gguf_dir> <out_csv>" >&2
    echo "  model_name   — base name before the quant suffix (e.g. 'olmoe', 'gemma-4-26B-A4B-it-Q4_K_M'-style glob root)" >&2
    echo "  gguf_dir     — directory holding the GGUF files" >&2
    echo "  out_csv      — path to write CSV (will be created/overwritten)" >&2
    echo "" >&2
    echo "Optional env vars:" >&2
    echo "  LLAMA_BIN_DIR        (default: /home/eisen/spectralai/llama.cpp/build/bin)" >&2
    echo "  WIKITEXT_PATH        (default: data/eval/wiki.test.raw)" >&2
    echo "  NGL                  (default: 99 — layers on GPU)" >&2
    echo "  PPL_CTX              (default: 512)" >&2
    echo "  PPL_CHUNKS           (default: 0 = full set)" >&2
    echo "  BENCH_REPS           (default: 3)" >&2
    echo "  BENCH_PROMPT_LEN     (default: 512)" >&2
    echo "  BENCH_GEN_LEN        (default: 128)" >&2
    exit 1
fi

MODEL_NAME="$1"
GGUF_DIR="$2"
OUT_CSV="$3"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-/home/eisen/spectralai/llama.cpp/build/bin}"
WIKITEXT_PATH="${WIKITEXT_PATH:-$PROJECT_ROOT/data/eval/wiki.test.raw}"
NGL="${NGL:-99}"
PPL_CTX="${PPL_CTX:-512}"
PPL_CHUNKS="${PPL_CHUNKS:-0}"
BENCH_REPS="${BENCH_REPS:-3}"
BENCH_PROMPT_LEN="${BENCH_PROMPT_LEN:-512}"
BENCH_GEN_LEN="${BENCH_GEN_LEN:-128}"

LLAMA_BENCH="$LLAMA_BIN_DIR/llama-bench"
LLAMA_PPL="$LLAMA_BIN_DIR/llama-perplexity"

for tool in "$LLAMA_BENCH" "$LLAMA_PPL"; do
    if [[ ! -x "$tool" ]]; then
        echo "ERROR: $tool not found or not executable" >&2
        exit 2
    fi
done

if [[ ! -f "$WIKITEXT_PATH" ]]; then
    echo "ERROR: WikiText-2 test file not found at $WIKITEXT_PATH" >&2
    echo "  Hint: run scripts/fetch_wikitext2.sh or extract via HF datasets" >&2
    exit 3
fi

# Prepare output CSV
mkdir -p "$(dirname "$OUT_CSV")"
echo "model,quant,size_gb,prefill_pp${BENCH_PROMPT_LEN}_tok_s,prefill_stdev,generate_tg${BENCH_GEN_LEN}_tok_s,generate_stdev,ppl_wikitext2_c${PPL_CTX},ppl_pm,prefill_tok_s_ppl,n_tokens_ppl" > "$OUT_CSV"

shopt -s nullglob
GGUFS=("$GGUF_DIR"/"$MODEL_NAME"*.gguf)

if [[ ${#GGUFS[@]} -eq 0 ]]; then
    echo "No GGUFs found matching '$GGUF_DIR/$MODEL_NAME*.gguf'" >&2
    exit 4
fi

echo "Found ${#GGUFS[@]} GGUF(s) for model '$MODEL_NAME'" >&2

for gguf in "${GGUFS[@]}"; do
    fname=$(basename "$gguf")
    # Derive quant label: strip model_name prefix and .gguf suffix
    quant="${fname#$MODEL_NAME-}"
    quant="${quant#$MODEL_NAME}"
    quant="${quant%.gguf}"
    # Drop a leading dash/underscore if any
    quant="${quant#-}"
    quant="${quant#_}"

    size_bytes=$(stat -c %s "$gguf")
    size_gb=$(awk -v b="$size_bytes" 'BEGIN{printf "%.2f", b/1024/1024/1024}')

    echo "=== $fname (quant=$quant, $size_gb GB) ===" >&2

    # llama-bench — emits 2 CSV rows (prompt, gen). We grab avg_ts + stddev_ts for each.
    # output csv has columns: ...,n_prompt,n_gen,...,avg_ns,stddev_ns,avg_ts,stddev_ts
    bench_csv=$("$LLAMA_BENCH" -m "$gguf" -p "$BENCH_PROMPT_LEN" -n "$BENCH_GEN_LEN" -ngl "$NGL" -r "$BENCH_REPS" --output csv 2>/dev/null || true)

    # Parse: pick the row where n_prompt=BENCH_PROMPT_LEN, n_gen=0 (prefill); and n_prompt=0, n_gen=BENCH_GEN_LEN (generate).
    pp_row=$(echo "$bench_csv" | awk -F',' -v p="$BENCH_PROMPT_LEN" '$0 ~ /".*","'"$BENCH_PROMPT_LEN"'","0"/' || true)
    # Simpler: the last two data rows from llama-bench are prefill + generate
    rows=$(echo "$bench_csv" | grep -E '^"' || true)
    pp_row=$(echo "$rows" | awk 'NR==1')
    tg_row=$(echo "$rows" | awk 'NR==2')

    # avg_ts is 4th from end, stddev_ts is 2nd from end in llama-bench CSV
    pp_avg=$(echo "$pp_row" | awk -F',' '{gsub(/"/,""); print $(NF-1)}')
    pp_std=$(echo "$pp_row" | awk -F',' '{gsub(/"/,""); print $NF}')
    tg_avg=$(echo "$tg_row" | awk -F',' '{gsub(/"/,""); print $(NF-1)}')
    tg_std=$(echo "$tg_row" | awk -F',' '{gsub(/"/,""); print $NF}')

    # llama-perplexity
    chunks_flag=""
    if [[ "$PPL_CHUNKS" != "0" ]]; then
        chunks_flag="--chunks $PPL_CHUNKS"
    fi

    ppl_out=$("$LLAMA_PPL" -m "$gguf" -f "$WIKITEXT_PATH" -c "$PPL_CTX" -b "$PPL_CTX" -ngl "$NGL" -t 8 $chunks_flag 2>&1)
    ppl_rc=$?
    if [[ $ppl_rc -ne 0 ]]; then
        echo "WARN: llama-perplexity exited $ppl_rc for $gguf" >&2
    fi

    ppl=$(echo "$ppl_out" | grep -oE 'Final estimate: PPL = [0-9.]+' 2>/dev/null | awk '{print $NF}')
    ppl_pm=$(echo "$ppl_out" | grep -oE 'Final estimate: PPL = [0-9.]+ \+/\- [0-9.]+' 2>/dev/null | awk '{print $NF}')
    ppl_prefill_tok_s=$(echo "$ppl_out" | grep -oE 'prompt eval time =.*tokens per second' 2>/dev/null | grep -oE '[0-9.]+ tokens per second' 2>/dev/null | awk '{print $1}')
    ppl_ntokens=$(echo "$ppl_out" | grep -oE 'prompt eval time = [0-9.]+ ms / [0-9]+ tokens' 2>/dev/null | awk '{print $(NF-1)}')

    row="$MODEL_NAME,$quant,$size_gb,${pp_avg:-NA},${pp_std:-NA},${tg_avg:-NA},${tg_std:-NA},${ppl:-NA},${ppl_pm:-NA},${ppl_prefill_tok_s:-NA},${ppl_ntokens:-NA}"
    echo "$row" | tee -a "$OUT_CSV"
done

echo "" >&2
echo "Wrote $OUT_CSV" >&2
