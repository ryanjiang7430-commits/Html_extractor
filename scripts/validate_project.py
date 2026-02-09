"""Project-level validation for extractor logic and Gradio demo."""

from __future__ import annotations

import json
from pathlib import Path

from app.gradio_app import SAMPLES, build_ui, extract_from_html, load_sample
from html_text_extractor import HtmlExtractor

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PATH = ROOT / "samples" / "samples.jsonl"


def load_samples() -> list[dict[str, str]]:
    with SAMPLES_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_extractor(samples: list[dict[str, str]]) -> tuple[int, int]:
    extractor = HtmlExtractor()
    non_empty = 0
    errors = 0
    for item in samples:
        try:
            main_text, title, titles = extractor.extract(item["html"])
            if not isinstance(main_text, str) or not isinstance(title, str) or not isinstance(titles, list):
                errors += 1
                continue
            if main_text.strip():
                non_empty += 1
        except Exception:
            errors += 1
    return non_empty, errors


def validate_gradio(samples: list[dict[str, str]]) -> None:
    demo = build_ui()
    if samples:
        title, main_text, titles_view, threshold_text, similarity_report = extract_from_html(
            samples[0]["html"]
        )
        assert isinstance(title, str)
        assert isinstance(main_text, str)
        assert isinstance(titles_view, str)
        assert isinstance(threshold_text, str)
        assert isinstance(similarity_report, str)

        choice = f"{0:02d} | {samples[0]['url']}"
        url, html, main_text, out_title = load_sample(choice)
        assert isinstance(url, str)
        assert isinstance(html, str)
        assert isinstance(main_text, str)
        assert isinstance(out_title, str)
    demo.close()


def main() -> None:
    samples = load_samples()
    assert 15 <= len(samples) <= 25, f"Unexpected sample size: {len(samples)}"
    assert len(SAMPLES) == len(samples), "Gradio samples are not loaded correctly."

    non_empty, errors = validate_extractor(samples)
    assert errors == 0, f"Extractor raised errors on {errors} sample(s)."
    assert non_empty >= len(samples) // 2, "Too many empty extraction results."

    validate_gradio(samples)
    print(
        "Validation passed.",
        f"samples={len(samples)}",
        f"non_empty={non_empty}",
        f"errors={errors}",
    )


if __name__ == "__main__":
    main()
