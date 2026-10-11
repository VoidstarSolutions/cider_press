#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Drive a running llama-server /v1/embeddings endpoint over a corpus in
omniscient's exact call shape and report throughput.

Batching mirrors omniscient `Embedder::embed_batched` (src/embed.rs): texts
are sent in order, a batch closes when it would exceed --max-batch-chunks
texts or --max-batch-bytes bytes, batches run serially. One warm-up batch
is sent before timing. Each timed run embeds the whole corpus.

Start the server the way omniscient's auto_start does (Qwen3-Embedding-0.6B
Q8_0, last-token pooling, ctx == batch == ubatch == 32000)::

    llama-server -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 --embedding \\
        --pooling last -c 32000 -b 32000 -ub 32000 --port 8089

Then::

    uv run scripts/bench_llama_embed.py --base-url http://127.0.0.1:8089 \\
        --corpus /tmp/embed-phase0/corpus.jsonl --runs 5 --self-check

Compare texts/s and bytes/s with `cider-press bench embed` (phase 3) or the
phase-0 proxy numbers; tokens/s is printed under the GGUF tokenizer only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def read_corpus(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, str):
                raise SystemExit(f"{path}:{lineno}: expected a JSON string")
            texts.append(value)
    return texts


def batches(texts: list[str], max_chunks: int, max_bytes: int):
    start, cur_bytes, cur_len = 0, 0, 0
    for i, t in enumerate(texts):
        n = len(t.encode("utf-8"))
        if cur_len > 0 and (cur_len >= max_chunks or cur_bytes + n > max_bytes):
            yield texts[start:i]
            start, cur_bytes, cur_len = i, 0, 0
        cur_bytes += n
        cur_len += 1
    if cur_len > 0:
        yield texts[start:]


def post_json(url: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {e.code} from {url}: {body[:400]}") from None


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def embed_batch(base: str, model: str, batch: list[str], check_norms: bool) -> int:
    """One request. The vector-count check is unconditional (a short response
    would silently inflate texts/s, as omniscient's own guard notes); the
    per-vector norm check is pure-Python work and runs only when asked, i.e.
    outside the timed passes."""
    out = post_json(f"{base}/v1/embeddings", {"model": model, "input": batch})
    data = out.get("data", [])
    if len(data) != len(batch):
        raise SystemExit(f"server returned {len(data)} vectors for a batch of {len(batch)}")
    if check_norms:
        for row in data:
            v = row["embedding"]
            norm = sum(x * x for x in v) ** 0.5
            if abs(norm - 1.0) > 1e-3:
                raise SystemExit(f"vector norm {norm:.4f} != 1 (expected normalised)")
    return len(data[0]["embedding"]) if data else 0


def count_tokens(base: str, texts: list[str]) -> int:
    total = 0
    for t in texts:
        total += len(post_json(f"{base}/tokenize", {"content": t}).get("tokens", []))
    return total


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8089")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0")
    p.add_argument("--corpus", required=True)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--max-batch-chunks", type=int, default=64)
    p.add_argument("--max-batch-bytes", type=int, default=32000)
    p.add_argument("--single-texts", type=int, default=50)
    p.add_argument("--self-check", action="store_true", help="also verify every warm-up vector is L2-normalised")
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    texts = read_corpus(Path(args.corpus))
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    props = get_json(f"{base}/props")
    tokens_gguf = count_tokens(base, texts)

    plan = list(batches(texts, args.max_batch_chunks, args.max_batch_bytes))
    dim = embed_batch(base, args.model, plan[0], args.self_check)  # warm-up; norms checked here only

    run_secs: list[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        for b in plan:
            embed_batch(base, args.model, b, False)
        run_secs.append(time.perf_counter() - t0)

    single_ms: list[float] = []
    for t in texts[: args.single_texts]:
        t0 = time.perf_counter()
        embed_batch(base, args.model, [t], False)
        single_ms.append((time.perf_counter() - t0) * 1e3)

    med = statistics.median(run_secs)
    report = {
        "engine": "llama-server",
        "model": args.model,
        "server_props": {k: props.get(k) for k in ("default_generation_settings", "total_slots", "model_path", "n_ctx")},
        "corpus": {"texts": len(texts), "bytes": total_bytes, "tokens_gguf": tokens_gguf, "batches": len(plan)},
        "dim": dim,
        "batched": {
            "runs": args.runs,
            "secs_median": med,
            "secs_min": min(run_secs),
            "secs_max": max(run_secs),
            "texts_per_s": len(texts) / med,
            "bytes_per_s": total_bytes / med,
            "tokens_gguf_per_s": tokens_gguf / med,
        },
        "single": {
            "texts": len(single_ms),
            "ms_median": statistics.median(single_ms) if single_ms else None,
            "ms_min": min(single_ms) if single_ms else None,
            "ms_max": max(single_ms) if single_ms else None,
        },
    }
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
