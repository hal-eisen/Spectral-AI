#!/usr/bin/env bash
# Fetch the WikiText-2 raw test split as wiki.test.raw for llama-perplexity.
# Requires the HF `datasets` package (available in the project's .venv).
set -euo pipefail

OUT="${1:-data/eval/wiki.test.raw}"
mkdir -p "$(dirname "$OUT")"

if [[ -s "$OUT" ]]; then
    echo "$OUT already exists ($(wc -l < "$OUT") lines)" >&2
    exit 0
fi

PYTHON="${PYTHON:-/home/eisen/spectralai/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "python not found at $PYTHON — set PYTHON= to your interpreter" >&2
    exit 1
fi

$PYTHON - <<PY
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
with open("$OUT", "w") as f:
    for row in ds:
        f.write(row["text"])
print(f"Wrote $OUT with {len(ds)} docs", flush=True)
PY
