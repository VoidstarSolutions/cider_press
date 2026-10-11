#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tokenizers>=0.20",
# ]
# ///
"""Build an omniscient-style embedding corpus from a repository, and cut
prompts of an exact token length from it.

`build` walks `git ls-files` of --repo, keeps UTF-8 text files, and emits
line windows capped at --max-bytes (omniscient splits oversized definitions
to fit its byte budget; the median chunk it sends is a few hundred bytes).
Output is ONE JSON STRING PER LINE so embedded newlines survive and both
engines read identical texts.

`prompt` concatenates corpus texts, tokenizes with the given tokenizer.json
(add_special_tokens=False, as cider's Tokenizer::encode does), truncates to
--tokens ids, decodes, and re-encodes to verify the round trip is exact;
it fails if the decoded text does not re-encode to exactly --tokens ids.

Usage::

    uv run scripts/make_embed_corpus.py build --repo ../omniscient \\
        --out /tmp/corpus.jsonl --texts 640
    uv run scripts/make_embed_corpus.py prompt \\
        --tokenizer-json $CIDER_PRESS_CHECKPOINT/tokenizer.json \\
        --tokens 150 --corpus /tmp/corpus.jsonl --out /tmp/p150.txt
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

MAX_BYTES_DEFAULT = 2000
WINDOW_LINES_DEFAULT = 20


def iter_text_files(repo: Path):
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    for rel in out.split(b"\0"):
        if not rel:
            continue
        path = repo / rel.decode("utf-8", "surrogateescape")
        if not path.is_file():
            continue
        try:
            yield path, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def chunks_from_text(text: str, max_bytes: int, window_lines: int):
    lines = text.splitlines()
    for start in range(0, len(lines), window_lines):
        window = "\n".join(lines[start : start + window_lines]).strip()
        if not window:
            continue
        encoded = window.encode("utf-8")
        while len(encoded) > max_bytes:
            cut = encoded[:max_bytes].decode("utf-8", "ignore")
            cut = cut.rsplit("\n", 1)[0] or cut
            yield cut
            window = window[len(cut) :].lstrip("\n")
            encoded = window.encode("utf-8")
        if window:
            yield window


def cmd_build(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    all_chunks = [
        c for _, text in iter_text_files(repo) for c in chunks_from_text(text, args.max_bytes, args.window_lines)
    ]
    if len(all_chunks) < args.texts:
        print(f"only {len(all_chunks)} chunks in {repo}; need {args.texts}", file=sys.stderr)
        return 1
    rng = random.Random(args.seed)
    chosen = rng.sample(all_chunks, args.texts)
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for c in chosen:
            f.write(json.dumps(c, ensure_ascii=False))
            f.write("\n")
    sizes = [len(c.encode("utf-8")) for c in chosen]
    print(
        f"texts={len(chosen)} bytes={sum(sizes)} mean_bytes={sum(sizes) // len(sizes)} "
        f"max_bytes={max(sizes)} -> {out}",
        file=sys.stderr,
    )
    return 0


def read_corpus(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, str):
                raise SystemExit(f"{path}:{lineno}: expected a JSON string, got {type(value).__name__}")
            texts.append(value)
    return texts


def cmd_prompt(args: argparse.Namespace) -> int:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer_json)
    texts = read_corpus(Path(args.corpus))
    joined = "\n".join(texts)
    ids = tok.encode(joined, add_special_tokens=False).ids
    if len(ids) < args.tokens:
        print(f"corpus has {len(ids)} tokens; need {args.tokens}", file=sys.stderr)
        return 1
    text = tok.decode(ids[: args.tokens], skip_special_tokens=False)
    n = len(tok.encode(text, add_special_tokens=False).ids)
    if n != args.tokens:
        print(f"round trip produced {n} tokens, wanted {args.tokens}", file=sys.stderr)
        return 1
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"tokens={n} bytes={len(text.encode('utf-8'))} -> {args.out}", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--repo", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--texts", type=int, default=640)
    b.add_argument("--max-bytes", type=int, default=MAX_BYTES_DEFAULT)
    b.add_argument("--window-lines", type=int, default=WINDOW_LINES_DEFAULT)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_build)
    q = sub.add_parser("prompt")
    q.add_argument("--tokenizer-json", required=True)
    q.add_argument("--tokens", type=int, required=True)
    q.add_argument("--corpus", required=True)
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_prompt)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
