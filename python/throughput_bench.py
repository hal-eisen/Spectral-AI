"""Generic tokens/sec benchmark harness.

Used by P1-6 (Gemma 4 tok/s row) and P3-6 (Qwen 3.6 tok/s row) to produce a
clean comparison between the unmodified HF forward pass, BVH-hybrid routing,
and BVH-pure routing. The measurement primitive here is intentionally narrow:
take any callable that consumes a token tensor and returns logits, and report
prefill tokens/sec and (optionally) autoregressive decode tokens/sec.

Keep this file HF-agnostic where possible. It accepts:
- a `torch.nn.Module` + tokenizer (for HF CausalLM models)
- or a plain `forward_fn(input_ids) -> logits` callable (for custom paths)

Writes a dict you can merge into results/tokens_per_sec_table.md.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import statistics
import time
from pathlib import Path
from typing import Callable, Optional

import torch


@dataclasses.dataclass
class ThroughputResult:
    label: str
    mode: str  # "prefill" or "generate"
    batch_size: int
    seq_len: int
    n_iters: int
    warmup_iters: int
    median_tok_per_sec: float
    mean_tok_per_sec: float
    stdev_tok_per_sec: float
    median_batch_latency_ms: float
    per_run_tok_per_sec: list[float]
    device: str
    dtype: str
    notes: str = ""

    def to_row(self) -> dict:
        return {
            "label": self.label,
            "mode": self.mode,
            "tok_s_median": round(self.median_tok_per_sec, 2),
            "tok_s_mean": round(self.mean_tok_per_sec, 2),
            "tok_s_stdev": round(self.stdev_tok_per_sec, 2),
            "latency_ms": round(self.median_batch_latency_ms, 2),
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "n_iters": self.n_iters,
            "device": self.device,
            "dtype": self.dtype,
            "notes": self.notes,
        }


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_random_ids(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


@torch.no_grad()
def measure_prefill_tok_per_sec(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    label: str,
    vocab_size: int,
    batch_size: int = 1,
    seq_len: int = 512,
    warmup_iters: int = 3,
    n_iters: int = 10,
    device: Optional[torch.device] = None,
    dtype_hint: str = "",
    notes: str = "",
    autocast_dtype: Optional[torch.dtype] = None,
) -> ThroughputResult:
    """Measure prefill throughput (tokens processed per second in one forward).

    `forward_fn` takes an input_ids tensor and must return logits (or anything;
    the return value is ignored). We count every input token as "processed".
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build inputs once
    input_ids = _make_random_ids(batch_size, seq_len, vocab_size, device)

    ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )

    # Warmup
    with ctx:
        for _ in range(warmup_iters):
            _ = forward_fn(input_ids)
    _sync_if_cuda()

    # Timed
    per_run_latency_s = []
    for _ in range(n_iters):
        _sync_if_cuda()
        t0 = time.perf_counter()
        with ctx:
            _ = forward_fn(input_ids)
        _sync_if_cuda()
        per_run_latency_s.append(time.perf_counter() - t0)

    tok_per_run = batch_size * seq_len
    per_run_tps = [tok_per_run / s for s in per_run_latency_s]

    return ThroughputResult(
        label=label,
        mode="prefill",
        batch_size=batch_size,
        seq_len=seq_len,
        n_iters=n_iters,
        warmup_iters=warmup_iters,
        median_tok_per_sec=statistics.median(per_run_tps),
        mean_tok_per_sec=statistics.fmean(per_run_tps),
        stdev_tok_per_sec=statistics.stdev(per_run_tps) if len(per_run_tps) > 1 else 0.0,
        median_batch_latency_ms=statistics.median(per_run_latency_s) * 1000,
        per_run_tok_per_sec=[round(x, 3) for x in per_run_tps],
        device=str(device),
        dtype=dtype_hint,
        notes=notes,
    )


@torch.no_grad()
def measure_generate_tok_per_sec(
    model,
    tokenizer,
    *,
    label: str,
    prompt: str = "The quick brown fox",
    batch_size: int = 1,
    max_new_tokens: int = 64,
    warmup_iters: int = 2,
    n_iters: int = 5,
    device: Optional[torch.device] = None,
    dtype_hint: str = "",
    notes: str = "",
) -> ThroughputResult:
    """Autoregressive decode tok/s via model.generate. HF-specific.

    Reports new tokens per second (not including prefill). This is the number
    end-users care about for interactive latency.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True).to(device)
    prefill_len = inputs.input_ids.shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    # Warmup
    for _ in range(warmup_iters):
        _ = model.generate(**inputs, **gen_kwargs)
    _sync_if_cuda()

    per_run_latency_s = []
    per_run_new_tokens = []
    for _ in range(n_iters):
        _sync_if_cuda()
        t0 = time.perf_counter()
        out = model.generate(**inputs, **gen_kwargs)
        _sync_if_cuda()
        elapsed = time.perf_counter() - t0
        new_tokens = (out.shape[1] - prefill_len) * batch_size
        per_run_latency_s.append(elapsed)
        per_run_new_tokens.append(new_tokens)

    per_run_tps = [n / s for n, s in zip(per_run_new_tokens, per_run_latency_s)]

    return ThroughputResult(
        label=label,
        mode="generate",
        batch_size=batch_size,
        seq_len=prefill_len + max_new_tokens,
        n_iters=n_iters,
        warmup_iters=warmup_iters,
        median_tok_per_sec=statistics.median(per_run_tps),
        mean_tok_per_sec=statistics.fmean(per_run_tps),
        stdev_tok_per_sec=statistics.stdev(per_run_tps) if len(per_run_tps) > 1 else 0.0,
        median_batch_latency_ms=statistics.median(per_run_latency_s) * 1000,
        per_run_tok_per_sec=[round(x, 3) for x in per_run_tps],
        device=str(device),
        dtype=dtype_hint,
        notes=notes,
    )


def write_results(results: list[ThroughputResult], path: Path) -> None:
    """Append-or-create a JSON results file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            existing = []
    existing.extend(r.to_row() for r in results)
    path.write_text(json.dumps(existing, indent=2))


def sanity_microbench() -> list[ThroughputResult]:
    """Run the harness on a toy model to prove it works without loading big HF weights.

    Used as a smoke test for the methodology itself — separate from running the
    real model benchmark which needs the safetensors on disk.
    """
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 32000
    hidden = 1024

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden)
            self.block = nn.Linear(hidden, hidden)
            self.head = nn.Linear(hidden, vocab_size, bias=False)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            x = torch.relu(self.block(x))
            return self.head(x)

    model = TinyLM().to(device).eval()

    def fwd(ids):
        return model(ids)

    results = []
    for seq_len in (128, 512):
        for batch in (1, 4):
            r = measure_prefill_tok_per_sec(
                fwd,
                label=f"tinylm_prefill_b{batch}_s{seq_len}",
                vocab_size=vocab_size,
                batch_size=batch,
                seq_len=seq_len,
                warmup_iters=3,
                n_iters=10,
                device=device,
                dtype_hint="fp32",
            )
            results.append(r)
            print(
                f"  {r.label}: median={r.median_tok_per_sec:.0f} tok/s "
                f"(stdev={r.stdev_tok_per_sec:.0f}, latency={r.median_batch_latency_ms:.2f}ms)"
            )
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true", help="Run toy-model smoke test")
    ap.add_argument("--out", type=Path, default=Path("results/throughput_sanity.json"))
    args = ap.parse_args()

    if args.sanity:
        print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
        results = sanity_microbench()
        write_results(results, args.out)
        print(f"Wrote {args.out}")
    else:
        print("Import this module and call measure_* functions. Use --sanity for a smoke test.")
