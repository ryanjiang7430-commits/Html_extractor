"""Gradio demo for HTML text extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from lxml.html import fromstring, etree

from html_text_extractor import HtmlExtractor
from html_text_extractor.simhash_utils import ComputeSimhash

SAMPLES_PATH = Path(__file__).resolve().parents[1] / "samples" / "samples.jsonl"
ORIGINAL_PROJECT_PATH = Path("/Volumes/External2T/INF/GitFiles/htmlextractor/html_extractor/udf")
DEFAULT_SCORE = HtmlExtractor.V4_DEFAULT_SCORE
DEFAULT_SUBTITLE_SCORE = HtmlExtractor.V4_DEFAULT_SUBTITLE_SCORE
DEFAULT_AVG_TEXT_LEN_MIN = HtmlExtractor.V4_DEFAULT_AVG_TEXT_LEN_MIN
DEFAULT_MAIN_THRESHOLD = HtmlExtractor.V4_DEFAULT_MAIN_THRESHOLD
DEFAULT_MAIN_LEN_FLOOR = HtmlExtractor.V4_DEFAULT_MAIN_LEN_FLOOR
DEFAULT_LINK_RATIO_MAX = HtmlExtractor.V4_DEFAULT_LINK_RATIO_MAX
DEFAULT_INTERWEAVE_TIME = HtmlExtractor.V4_DEFAULT_INTERWEAVE_TIME
DEFAULT_FOOTER_PENALTY_RATIO = HtmlExtractor.V4_FOOTER_PENALTY_RATIO


def load_samples() -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    if not SAMPLES_PATH.exists():
        return samples
    with SAMPLES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            samples.append(item)
    return samples


extractor = HtmlExtractor()
legacy_extractor: Optional[object] = None
legacy_compute_simhash_cls: Optional[type] = None
legacy_extractor_error: str = ""
SAMPLES = load_samples()
SAMPLE_CHOICES = [f"{idx:02d} | {item['url']}" for idx, item in enumerate(SAMPLES)]


def get_legacy_extractor() -> Tuple[Optional[object], str]:
    """Lazy-load the original project extractor."""
    global legacy_extractor, legacy_compute_simhash_cls, legacy_extractor_error
    if legacy_extractor is not None:
        return legacy_extractor, ""
    if legacy_extractor_error:
        return None, legacy_extractor_error
    if not ORIGINAL_PROJECT_PATH.exists():
        legacy_extractor_error = f"Path not found: {ORIGINAL_PROJECT_PATH}"
        return None, legacy_extractor_error

    try:
        import emoji

        if not hasattr(emoji, "UNICODE_EMOJI") and hasattr(emoji, "EMOJI_DATA"):
            emoji.UNICODE_EMOJI = {"en": emoji.EMOJI_DATA}
        sys.path.insert(0, str(ORIGINAL_PROJECT_PATH))
        from compute_simhash2 import ComputeSimhash as LegacyComputeSimhash
        from html_extractor import HtmlExtractor as LegacyHtmlExtractor

        legacy_compute_simhash_cls = LegacyComputeSimhash
        legacy_extractor = LegacyHtmlExtractor()
        return legacy_extractor, ""
    except Exception as exc:  # pragma: no cover - demo fallback path
        legacy_extractor_error = f"{type(exc).__name__}: {exc}"
        return None, legacy_extractor_error


def render_titles(titles: list) -> str:
    """Render title candidate list as plain text."""
    return "\n".join([f"{score:.3f}\t{title_text}" for score, _, title_text in titles])


def build_threshold_text(score: int, subtitle_score: float) -> str:
    """Return current threshold settings used by extraction."""
    return (
        f"simhash_distance_threshold={score}\n"
        f"subtitle_similarity_threshold={subtitle_score}\n"
        f"avg_text_len_min={DEFAULT_AVG_TEXT_LEN_MIN}\n"
        f"main_threshold={DEFAULT_MAIN_THRESHOLD}\n"
        f"main_len_floor={DEFAULT_MAIN_LEN_FLOOR}\n"
        f"link_ratio_max={DEFAULT_LINK_RATIO_MAX}\n"
        f"interweave_time={DEFAULT_INTERWEAVE_TIME}\n"
        f"footer_penalty_ratio={DEFAULT_FOOTER_PENALTY_RATIO}\n"
        "tag_similarity = 1 - distance/128"
    )


def build_tag_similarity_report(html: str, score: int) -> str:
    """Build similarity report using merged legacy-like V4 behavior."""
    if not html:
        return ""
    probe = HtmlExtractor()
    html = probe.normalize_html_input(html)
    if not html:
        return ""
    try:
        element = fromstring(html)
    except Exception as exc:
        return f"[report parse error] {type(exc).__name__}: {exc}"

    etree.strip_elements(element, *probe.CONTENT_EXTRACTOR_USELESS_TAGS)
    etree.strip_tags(element, *probe.CONTENT_EXTRACTOR_STRIP_TAGS)
    probe.remove_children(element, probe.CONTENT_EXTRACTOR_NOISE_XPATHS)

    elements = element.xpath("//body")
    if not elements:
        return ""

    descendants = list(probe.descendants(elements[0], True))
    for node in descendants:
        if (node.text and node.text.strip() != "") or (node.tail and node.tail.strip() != ""):
            probe.rebuild_tree(node)

    elements = element.xpath("//body")
    descendants = list(probe.descendants(elements[0], True))
    simhash = ComputeSimhash()
    text_dict: dict[str, dict] = {}
    text_index = 0
    for node in descendants:
        if (node.text and node.text.strip() != "") or (node.tail and node.tail.strip() != ""):
            key = probe.get_tag_key(node)
            text = node.text.strip() if node.text else ""
            tail = node.tail.strip() if node.tail else ""
            merged_text = text + tail
            is_new = True
            key_hash = simhash.evaluate(key)
            for existing_key in text_dict:
                existing_hash = simhash.evaluate(existing_key)
                if simhash.distance(key_hash, existing_hash) <= score:
                    text_dict[existing_key]["text"] = "\n".join(
                        [text_dict[existing_key]["text"], merged_text]
                    )
                    text_dict[existing_key]["index"].append(text_index)
                    if node.tag == "a":
                        text_dict[existing_key]["atext_len"] += len(probe.trim(merged_text) or "")
                    is_new = False
            if is_new:
                text_dict[key] = {
                    "text": merged_text,
                    "index": [text_index],
                    "atext_len": len(probe.trim(merged_text) or "") if node.tag == "a" else 0,
                }
            text_index += 1

    if not text_dict:
        return ""

    max_length = 0
    max_key = ""
    for key in text_dict:
        trimmed = probe.trim(text_dict[key]["text"]) or ""
        effective_length = len(trimmed)
        if probe.has_footer_noise_keywords(text_dict[key]["text"]):
            effective_length = int(effective_length * DEFAULT_FOOTER_PENALTY_RATIO)
        if (
            max_length < effective_length
            and text_dict[key]["atext_len"] / max(1, len(trimmed)) < DEFAULT_LINK_RATIO_MAX
            and probe.get_avg_text_len(text_dict[key]["text"]) >= DEFAULT_AVG_TEXT_LEN_MIN
        ):
            max_length = effective_length
            max_key = key

    if not max_key:
        return ""

    main_text_keys = probe.get_main_text_keys(
        max_key,
        text_dict,
        threshold=DEFAULT_MAIN_THRESHOLD,
        main_len_floor=DEFAULT_MAIN_LEN_FLOOR,
        link_ratio_max=DEFAULT_LINK_RATIO_MAX,
        avg_text_len_min=DEFAULT_AVG_TEXT_LEN_MIN,
        interweave_time=DEFAULT_INTERWEAVE_TIME,
    )
    max_hash = simhash.evaluate(max_key)
    rows = []
    for key, value in text_dict.items():
        key_hash = simhash.evaluate(key)
        distance = simhash.distance(key_hash, max_hash)
        similarity = 1.0 - float(distance) / 128.0
        text_len = len(probe.trim(value["text"]) or "")
        is_main = probe.is_main_text_key(key, score, main_text_keys)
        rows.append((distance, similarity, key, text_len, "Y" if is_main else "N"))
    rows.sort(key=lambda x: (x[0], -x[3]))

    lines = [
        f"主结构key: {max_key}",
        "列: mainTextKey(左侧实际判定)\tdistance_to_main\tsimilarity\ttext_len\ttag_key",
    ]
    for distance, similarity, key, text_len, is_main in rows[:120]:
        lines.append(f"{is_main}\t{distance}\t{similarity:.3f}\t{text_len}\t{key}")
    return "\n".join(lines)


def build_legacy_tag_similarity_report(html: str, score: int) -> str:
    """Build similarity report following legacy UDF V4 behavior."""
    if not html:
        return ""
    legacy, error = get_legacy_extractor()
    if legacy is None:
        return f"[legacy load error] {error}"
    if legacy_compute_simhash_cls is None:
        return "[legacy load error] compute_simhash2 is not available"

    try:
        element = fromstring(html)
    except Exception as exc:
        return f"[legacy report parse error] {type(exc).__name__}: {exc}"

    etree.strip_elements(element, *legacy.CONTENT_EXTRACTOR_USELESS_TAGS)
    etree.strip_tags(element, *legacy.CONTENT_EXTRACTOR_STRIP_TAGS)
    legacy.remove_children(element, legacy.CONTENT_EXTRACTOR_NOISE_XPATHS)

    elements = element.xpath("//body")
    if not elements:
        return ""

    descendants = list(legacy.descendants(elements[0], True))
    for node in descendants:
        if (node.text and node.text.strip() != "") or (node.tail and node.tail.strip() != ""):
            legacy.rebuild_tree(node)

    elements = element.xpath("//body")
    descendants = list(legacy.descendants(elements[0], True))
    simhash = legacy_compute_simhash_cls()
    text_dict: dict[str, dict] = {}
    text_index = 0
    for node in descendants:
        if (node.text and node.text.strip() != "") or (node.tail and node.tail.strip() != ""):
            type_key = legacy.get_tagKey(node)
            text = node.text.strip() if node.text else ""
            tail = node.tail.strip() if node.tail else ""
            merged_text = text + tail
            is_new = True
            type_key_hash = simhash.evaluate(type_key)
            for key in text_dict:
                key_hash = simhash.evaluate(key)
                if simhash.distance(type_key_hash, key_hash) <= score:
                    text_dict[key]["text"] = "\n".join([text_dict[key]["text"], merged_text])
                    text_dict[key]["index"].append(text_index)
                    is_new = False
                    if node.tag == "a":
                        text_dict[key]["atext_len"] += len(legacy.trim(merged_text) or "")
            if is_new:
                text_dict[type_key] = {
                    "text": merged_text,
                    "index": [text_index],
                    "atext_len": len(legacy.trim(merged_text) or "") if node.tag == "a" else 0,
                }
            text_index += 1

    if not text_dict:
        return ""

    max_length = 0
    max_key = ""
    for key in text_dict:
        trimmed = legacy.trim(text_dict[key]["text"]) or ""
        if (
            max_length < len(trimmed)
            and text_dict[key]["atext_len"] / max(1, len(trimmed)) < 0.4
            and legacy.get_avgTextLen(text_dict[key]["text"]) >= 20
        ):
            max_length = len(trimmed)
            max_key = key

    if not max_key:
        return ""

    main_text_keys = legacy.get_mainText_keys(max_key, text_dict)
    max_hash = simhash.evaluate(max_key)
    rows = []
    for key, value in text_dict.items():
        key_hash = simhash.evaluate(key)
        distance = simhash.distance(key_hash, max_hash)
        similarity = 1.0 - float(distance) / 128.0
        text_len = len(legacy.trim(value["text"]) or "")
        is_main = legacy.is_mainText_key(key, main_text_keys, score)
        rows.append((distance, similarity, key, text_len, "Y" if is_main else "N"))
    rows.sort(key=lambda x: (x[0], -x[3]))

    lines = [
        f"主结构key(legacy): {max_key}",
        "列: mainTextKey\tdistance_to_main\tsimilarity\ttext_len\ttag_key",
    ]
    for distance, similarity, key, text_len, is_main in rows[:120]:
        lines.append(f"{is_main}\t{distance}\t{similarity:.3f}\t{text_len}\t{key}")
    return "\n".join(lines)


def extract_from_html(html: str):
    if not html:
        return "", "", "", "", ""
    main_text, title, titles = extractor.extract(html, DEFAULT_SCORE, DEFAULT_SUBTITLE_SCORE)
    titles_view = render_titles(titles)
    threshold_text = build_threshold_text(DEFAULT_SCORE, DEFAULT_SUBTITLE_SCORE)
    similarity_report = build_tag_similarity_report(html, DEFAULT_SCORE)
    return title, main_text, titles_view, threshold_text, similarity_report


def extract_from_html_legacy(html: str):
    """Run extraction via the original project directory implementation."""
    if not html:
        return "", "", "", ""
    legacy, error = get_legacy_extractor()
    if legacy is None:
        return "", "", f"[legacy load error] {error}", f"[legacy load error] {error}"
    try:
        main_result = legacy.htmlExtractor_V4(html, 15, 0.8)
        if isinstance(main_result, tuple):
            main_text = main_result[0] if len(main_result) >= 1 else ""
            titles = main_result[2] if len(main_result) >= 3 and isinstance(main_result[2], list) else []
        elif isinstance(main_result, str):
            # UDF version returns plain main text string.
            main_text = main_result
            titles = []
        else:
            main_text = ""
            titles = []

        title = ""
        if hasattr(legacy, "titleExtractor"):
            title_result = legacy.titleExtractor(html)
            title = title_result[0] if isinstance(title_result, tuple) else ""
        titles_view = render_titles(titles if isinstance(titles, list) else [])
        similarity_report = build_legacy_tag_similarity_report(html, DEFAULT_SCORE)
        return title, main_text, titles_view, similarity_report
    except Exception as exc:
        error_text = f"[legacy run error] {type(exc).__name__}: {exc}"
        return "", "", error_text, error_text


def extract_both(html: str):
    """Return current and legacy extraction results for UI display."""
    title, main_text, titles_view, threshold_text, similarity_report = extract_from_html(html)
    legacy_title, legacy_main_text, legacy_titles_view, legacy_similarity_report = extract_from_html_legacy(html)
    return (
        title,
        main_text,
        titles_view,
        threshold_text,
        similarity_report,
        legacy_title,
        legacy_main_text,
        legacy_titles_view,
        legacy_similarity_report,
    )


def load_sample(choice: str):
    if not choice:
        return "", "", "", ""
    idx = int(choice.split("|")[0].strip())
    sample = SAMPLES[idx]
    title, main_text, _, _, _ = extract_from_html(sample["html"])
    return sample["url"], sample["html"], main_text, title


def load_sample_with_compare(choice: str):
    if not choice:
        return "", "", "", "", "", "", "", "", "", "", ""
    idx = int(choice.split("|")[0].strip())
    sample = SAMPLES[idx]
    (
        title,
        main_text,
        titles_view,
        threshold_text,
        similarity_report,
        legacy_title,
        legacy_main_text,
        legacy_titles_view,
        legacy_similarity_report,
    ) = extract_both(sample["html"])
    return (
        sample["url"],
        sample["html"],
        title,
        main_text,
        titles_view,
        threshold_text,
        similarity_report,
        legacy_title,
        legacy_main_text,
        legacy_titles_view,
        legacy_similarity_report,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="HTML Text Extractor") as demo:
        gr.Markdown(
            "# HTML Text Extractor\n"
            "选择样本或粘贴HTML，查看当前实现与原始目录逻辑（legacy）的正文提取对比。\n"
            "当前实现主线已合并右侧算法，并启用参数（score=12, avg_text_len_min=10）。\n"
            f"legacy来源: `{ORIGINAL_PROJECT_PATH}`，调用算法: `udf.htmlExtractor_V4(score=15, subtitle_score=0.8)`。"
        )
        with gr.Tab("样本"):
            dropdown = gr.Dropdown(
                label="选择样本",
                choices=SAMPLE_CHOICES,
                value=SAMPLE_CHOICES[0] if SAMPLE_CHOICES else None,
            )
            url = gr.Textbox(label="URL", interactive=False)
            html = gr.Textbox(label="HTML", lines=12)
            with gr.Row():
                with gr.Column():
                    title = gr.Textbox(label="当前实现标题")
                    main_text = gr.Textbox(label="当前实现正文", lines=10)
                    titles_view = gr.Textbox(label="当前实现标题候选(相似度\t文本)", lines=6)
                    threshold_view = gr.Textbox(label="当前阈值设置", lines=9)
                    similarity_view = gr.Textbox(
                        label="标签组合相似度报告(当前实现)",
                        lines=12,
                    )
                with gr.Column():
                    legacy_title = gr.Textbox(label="原始逻辑标题")
                    legacy_main_text = gr.Textbox(label="原始逻辑正文", lines=10)
                    legacy_titles_view = gr.Textbox(label="原始逻辑标题候选(相似度\t文本)", lines=6)
                    legacy_similarity_view = gr.Textbox(
                        label="标签组合相似度报告(原始逻辑)",
                        lines=12,
                    )
            dropdown.change(
                load_sample_with_compare,
                inputs=dropdown,
                outputs=[
                    url,
                    html,
                    title,
                    main_text,
                    titles_view,
                    threshold_view,
                    similarity_view,
                    legacy_title,
                    legacy_main_text,
                    legacy_titles_view,
                    legacy_similarity_view,
                ],
            )
            if SAMPLE_CHOICES:
                demo.load(
                    load_sample_with_compare,
                    inputs=dropdown,
                    outputs=[
                        url,
                        html,
                        title,
                        main_text,
                        titles_view,
                        threshold_view,
                        similarity_view,
                        legacy_title,
                        legacy_main_text,
                        legacy_titles_view,
                        legacy_similarity_view,
                    ],
                )

        with gr.Tab("自定义"):
            html_input = gr.Textbox(label="HTML", lines=14)
            with gr.Row():
                with gr.Column():
                    title_output = gr.Textbox(label="当前实现标题")
                    main_output = gr.Textbox(label="当前实现正文", lines=10)
                    titles_output = gr.Textbox(label="当前实现标题候选(相似度\t文本)", lines=6)
                    threshold_output = gr.Textbox(label="当前阈值设置", lines=9)
                    similarity_output = gr.Textbox(
                        label="标签组合相似度报告(当前实现)",
                        lines=12,
                    )
                with gr.Column():
                    legacy_title_output = gr.Textbox(label="原始逻辑标题")
                    legacy_main_output = gr.Textbox(label="原始逻辑正文", lines=10)
                    legacy_titles_output = gr.Textbox(
                        label="原始逻辑标题候选(相似度\t文本)",
                        lines=6,
                    )
                    legacy_similarity_output = gr.Textbox(
                        label="标签组合相似度报告(原始逻辑)",
                        lines=12,
                    )
            run = gr.Button("提取")
            run.click(
                extract_both,
                inputs=html_input,
                outputs=[
                    title_output,
                    main_output,
                    titles_output,
                    threshold_output,
                    similarity_output,
                    legacy_title_output,
                    legacy_main_output,
                    legacy_titles_output,
                    legacy_similarity_output,
                ],
            )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
