"""Build a small sample JSONL from a large HTML dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional


def strip_http_headers(text: str) -> str:
    if not text:
        return ""
    if text.lstrip().startswith("<"):
        return text
    idx = text.find("<")
    return text[idx:] if idx != -1 else text


def read_samples(path: Path, limit: int) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            payload = raw
            if "jsonl" in raw:
                payload = json.loads(raw["jsonl"])
            html = strip_http_headers(payload.get("text", ""))
            url = payload.get("url", "")
            if not html:
                continue
            yield {"url": url, "html": html}
            limit -= 1
            if limit <= 0:
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as f:
        for item in read_samples(args.input, args.limit):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
