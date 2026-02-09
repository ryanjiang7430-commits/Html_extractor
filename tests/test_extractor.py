"""Regression tests for HtmlExtractor with local sample pages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from html_text_extractor import HtmlExtractor

SAMPLES_PATH = Path(__file__).resolve().parents[1] / "samples" / "samples.jsonl"


def _load_samples() -> list[dict[str, Any]]:
    with SAMPLES_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_sample_size_is_about_20() -> None:
    samples = _load_samples()
    assert 15 <= len(samples) <= 25
    assert all("url" in item and "html" in item for item in samples)


def test_extract_runs_on_all_samples_without_exceptions() -> None:
    extractor = HtmlExtractor()
    samples = _load_samples()
    failed_urls: list[str] = []
    non_empty_count = 0

    for item in samples:
        try:
            main_text, title, titles = extractor.extract(item["html"])
        except Exception:
            failed_urls.append(item["url"])
            continue

        assert isinstance(main_text, str)
        assert isinstance(title, str)
        assert isinstance(titles, list)
        if main_text.strip():
            non_empty_count += 1

    assert not failed_urls
    assert non_empty_count >= len(samples) // 2


def test_extract_is_stable_across_multiple_rounds() -> None:
    extractor = HtmlExtractor()
    samples = _load_samples()
    first_round = [extractor.extract_main_text(item["html"]) for item in samples]
    second_round = [extractor.extract_main_text(item["html"]) for item in samples]
    assert first_round == second_round
