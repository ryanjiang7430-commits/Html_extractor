"""HTML main-text and title extraction."""

from __future__ import annotations

import hashlib
import math
import string
from typing import List, Optional, Sequence

import langdetect
import numpy as np
import regex as re
from lxml.html import HtmlElement, etree, fromstring

from html_text_extractor.simhash_utils import ComputeSimhash
from html_text_extractor.stopwords import stopwords

_XML_INVALID_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

try:  # emoji data changed across versions
    from emoji import UNICODE_EMOJI

    def _emoji_set() -> set[str]:
        data = UNICODE_EMOJI
        if isinstance(data, dict):
            data = data.get("en", data)
        return set(data.keys())

except Exception:  # pragma: no cover - fallback for new emoji versions
    from emoji import EMOJI_DATA

    def _emoji_set() -> set[str]:
        return set(EMOJI_DATA.keys())


class HtmlExtractor:
    """Extract main text and titles from raw HTML."""

    CONTENT_EXTRACTOR_USELESS_TAGS = [
        "meta",
        "style",
        "script",
        "link",
        "video",
        "audio",
        "iframe",
        "source",
        "svg",
        "path",
        "symbol",
        "img",
        "footer",
        "select",
        "option",
        "nav",
        "marquee",
    ]
    CONTENT_EXTRACTOR_STRIP_TAGS = ["span", "blockquote", "strong"]
    CONTENT_EXTRACTOR_NOISE_XPATHS = [
        '//*[contains(@class, "advertisement")]',
        '//*[contains(@class, "advert")]',
        '//*[contains(@style, "display: none")]',
        '//*[contains(@style, "display:none")]',
        "//comment()",
    ]
    METAS = [
        '//meta[starts-with(@property, "og:title")]/@content',
        '//meta[starts-with(@name, "og:title")]/@content',
        '//meta[starts-with(@property, "title")]/@content',
        '//meta[starts-with(@name, "title")]/@content',
        '//meta[starts-with(@property, "page:title")]/@content',
    ]
    V4_DEFAULT_SCORE = 12
    V4_DEFAULT_SUBTITLE_SCORE = 0.8
    V4_DEFAULT_AVG_TEXT_LEN_MIN = 10
    V4_DEFAULT_MAIN_THRESHOLD = 0.7
    V4_DEFAULT_MAIN_LEN_FLOOR = 200
    V4_DEFAULT_LINK_RATIO_MAX = 0.3
    V4_DEFAULT_INTERWEAVE_TIME = 2
    V4_FOOTER_PENALTY_RATIO = 1.0
    FOOTER_NOISE_KEYWORDS = (
        "copyright",
        "版权",
        "版權",
        "免责声明",
        "免責聲明",
        "备案",
        "icp",
        "公网安备",
    )

    def __init__(self) -> None:
        self.simhash: Optional[ComputeSimhash] = None
        self.lang: str = ""

    def sanitize_xml_text(self, text: str) -> str:
        """Remove control characters that are invalid in XML/HTML nodes."""
        return _XML_INVALID_CHARS_RE.sub("", text)

    def normalize_html_input(self, html: str) -> str:
        """Normalize HTML input for robust parsing."""
        if not html:
            return ""
        html = self.sanitize_xml_text(html)
        html = html.replace("&nbsp;", "")
        stripped = html.lstrip()
        if stripped and stripped[0] != "<":
            first_tag = html.find("<")
            if first_tag == -1:
                return ""
            html = html[first_tag:]
        return html

    def has_footer_noise_keywords(self, text: str) -> bool:
        """Return True when text resembles footer/legal boilerplate."""
        lowered = text.lower()
        return any(token in lowered for token in self.FOOTER_NOISE_KEYWORDS)

    def remove_element(self, element: Optional[HtmlElement]) -> None:
        """Remove an element from its parent if it exists."""
        if element is None:
            return
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)

    def remove_children(
        self, element: Optional[HtmlElement], xpaths: Sequence[str]
    ) -> Optional[HtmlElement]:
        """Remove children matching any XPath from an element."""
        if element is None or not xpaths:
            return element
        for xpath in xpaths:
            nodes = element.xpath(xpath)
            for node in nodes:
                self.remove_element(node)
        return element

    def descendants(self, element: Optional[HtmlElement], including: bool = False):
        """Yield descendants of an element."""
        if element is None:
            return []
        if including:
            yield element
        for descendant in element.iterdescendants():
            yield descendant

    def rebuild_tree(self, node: HtmlElement) -> None:
        """Flatten nested nodes into a single text node."""
        if node.text:
            node.text = self.sanitize_xml_text(node.text)
        if node.tail:
            node.tail = self.sanitize_xml_text(node.tail)
        if (not node.text or node.text.rstrip() == "") and (
            not node.tail or node.tail.rstrip() == ""
        ):
            return
        if node.getchildren():
            for child in node.getchildren():
                if child.tag == "br":
                    if node.text:
                        node.text += "\n"
                    else:
                        node.text = "\n"
                if (child.text and child.text.rstrip() != "") or (
                    child.tail and child.tail.rstrip() != ""
                ):
                    self.rebuild_tree(child)
                    text = self.sanitize_xml_text(child.text or "")
                    tail = self.sanitize_xml_text(child.tail or "")
                    if child.tag in {"div", "p", "h1", "h2", "h3", "h4", "h5", "h6"}:
                        text = f"\n{text}\n"
                    if node.text:
                        node.text += text + tail
                    else:
                        node.text = text + tail
                    for grandchild in child:
                        node.insert(node.index(child), grandchild)
                    node.remove(child)

    def trim(self, value: Optional[str]) -> Optional[str]:
        """Trim all whitespace from a string."""
        try:
            return "".join(value.split()).strip()
        except (AttributeError, TypeError):
            return None

    def clean_main_text(self, main_text: str) -> str:
        """Deduplicate and clean extracted paragraphs."""
        paragraphs = main_text.split("\n")
        cleaned_main_text = ""
        md5_set = set()
        for paragraph in paragraphs:
            trimmed = self.trim(paragraph) or ""
            md5 = hashlib.md5(trimmed.encode("utf-8")).hexdigest()
            if md5 not in md5_set:
                md5_set.add(md5)
                if trimmed != "":
                    cleaned_main_text += paragraph.rstrip().lstrip() + "\n"
        if cleaned_main_text.endswith("\n"):
            cleaned_main_text = cleaned_main_text[:-1]
        return cleaned_main_text

    def html_extractor(self, line: str) -> str:
        """Extract main text using the original heuristic version (V1)."""
        try:
            element = fromstring(line)
        except Exception:
            return ""

        etree.strip_elements(element, *self.CONTENT_EXTRACTOR_USELESS_TAGS)
        etree.strip_tags(element, *self.CONTENT_EXTRACTOR_STRIP_TAGS)
        self.remove_children(element, self.CONTENT_EXTRACTOR_NOISE_XPATHS)

        elements = element.xpath("//body")
        if not elements:
            return ""

        descendants_list = list(self.descendants(elements[0], True))
        text_dict: dict[str, str] = {}
        for descendant in descendants_list:
            if (descendant.text and descendant.text.rstrip() != "") or (
                descendant.tail and descendant.tail.rstrip() != ""
            ):
                self.rebuild_tree(descendant)
                tag = descendant.tag.lower() if isinstance(descendant.tag, str) else ""
                css_class = "".join(descendant.xpath("./@class")).lower()
                css_style = "".join(descendant.xpath("./@style")).lower()
                type_key = "\\|".join([tag, css_class, css_style])
                text = descendant.text or ""
                tail = descendant.tail or ""
                if len((text + tail).replace(" ", "").replace("\t", "").replace("\n", "")) >= 120:
                    if type_key not in text_dict:
                        text_dict[type_key] = text + tail
                    else:
                        text_dict[type_key] = "\n".join([text_dict[type_key], text + tail])

        max_length = 0
        max_key = ""
        for type_key, value in text_dict.items():
            value_len = len(value.replace(" ", "").replace("\t", "").replace("\n", ""))
            if max_length < value_len:
                max_length = value_len
                max_key = type_key
        return text_dict.get(max_key, "") if max_key else ""

    def get_tagpath(self, node: HtmlElement) -> str:
        """Return tagpath for a node."""
        tagpath = ""
        for ancestor in node.iterancestors():
            if ancestor.tag is not None:
                tagpath = ancestor.tag.lower() + "/" + tagpath
            else:
                tagpath += "/"
        if node.tag is not None and isinstance(node.tag, str):
            tagpath += node.tag.lower()
        return tagpath

    def html_extractor_v2(self, line: str, score: int):
        """Extract main text with Simhash grouping (V2)."""
        simhash = ComputeSimhash()
        try:
            element = fromstring(line)
        except Exception:
            return ""

        etree.strip_elements(element, *self.CONTENT_EXTRACTOR_USELESS_TAGS)
        etree.strip_tags(element, *self.CONTENT_EXTRACTOR_STRIP_TAGS)
        self.remove_children(element, self.CONTENT_EXTRACTOR_NOISE_XPATHS)

        elements = element.xpath("//body")
        if not elements:
            return ""

        descendants_list = list(self.descendants(elements[0], True))
        for descendant in descendants_list:
            if (descendant.text and descendant.text.rstrip() != "") or (
                descendant.tail and descendant.tail.rstrip() != ""
            ):
                self.rebuild_tree(descendant)

        elements = element.xpath("//body")
        descendants_list = list(self.descendants(elements[0], True))
        text_dict: dict[str, str] = {}
        for descendant in descendants_list:
            if (descendant.text and descendant.text.rstrip() != "") or (
                descendant.tail and descendant.tail.rstrip() != ""
            ):
                tagpath = self.get_tagpath(descendant).replace("|", "")
                css_class = "".join(descendant.xpath("./@class")).lower().replace("|", "")
                css_style = "".join(descendant.xpath("./@style")).lower().replace("|", "")
                type_key = tagpath + "|" + css_class + "|" + css_style
                text = descendant.text or ""
                tail = descendant.tail or ""
                is_new = True
                type_key_hash = simhash.evaluate(type_key)
                for key in text_dict:
                    key_hash = simhash.evaluate(key)
                    if simhash.distance(type_key_hash, key_hash) <= score:
                        text_dict[key] = "\n".join([text_dict[key], text + tail])
                        is_new = False
                if is_new:
                    text_dict[type_key] = text + tail

        max_length = 0
        max_key = ""
        for type_key, value in text_dict.items():
            value_len = len(value.replace(" ", "").replace("\t", "").replace("\n", ""))
            if max_length < value_len:
                max_length = value_len
                max_key = type_key

        if not max_key:
            return ""

        exclude_list = []
        include_list = []
        max_key_hash = simhash.evaluate(max_key)
        for descendant in descendants_list:
            if (descendant.text and descendant.text.rstrip() != "") or (
                descendant.tail and descendant.tail.rstrip() != ""
            ):
                tagpath = self.get_tagpath(descendant).replace("|", "")
                css_class = "".join(descendant.xpath("./@class")).lower().replace("|", "")
                css_style = "".join(descendant.xpath("./@style")).lower().replace("|", "")
                type_key = tagpath + "|" + css_class + "|" + css_style
                type_key_hash = simhash.evaluate(type_key)
                text = descendant.text or ""
                tail = descendant.tail or ""
                distance = simhash.distance(type_key_hash, max_key_hash)
                if distance > score:
                    exclude_list.append([str(distance), type_key, text + tail])
                else:
                    include_list.append([str(distance), type_key, text + tail])
        return text_dict[max_key], max_key, exclude_list, include_list

    def extract_by_meta(self, element: HtmlElement) -> Optional[str]:
        """Extract title from meta tags."""
        for xpath in self.METAS:
            title = element.xpath(xpath)
            if title:
                return "".join(title)
        return None

    def extract_by_title(self, element: HtmlElement) -> str:
        """Extract title from <title> tag."""
        return "".join(element.xpath("//title//text()")).strip()

    def extract_by_hs(self, element: HtmlElement) -> List[str]:
        """Extract all h1-h3 texts."""
        hs = element.xpath("//h1//text()|//h2//text()|//h3//text()")
        return hs or []

    def extract_by_h(self, element: HtmlElement) -> Optional[str]:
        """Extract the first h1/h2/h3 text."""
        for xpath in ["//h1", "//h2", "//h3"]:
            children = element.xpath(xpath)
            if not children:
                continue
            child = children[0]
            texts = child.xpath("./text()")
            if texts and len(texts):
                return texts[0].strip()
        return None

    def zh_tokenize(self, text: str) -> List[str]:
        """Tokenize Chinese text while filtering stopwords and symbols."""
        import jieba

        main_special_characters = string.punctuation + string.digits + string.whitespace
        other_special_characters = (
            "    　    ￼’“”–ー一▬…✦�­£​•€«»°·═"
            "×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖"
            "◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚"
            "゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖"
            "」﴾》"
        )
        emoji_chars = list(_emoji_set())
        special_characters_default = set(main_special_characters + other_special_characters)
        special_characters_default.update(emoji_chars)
        stop_words = set(stopwords["zh"]) | special_characters_default
        token_result = jieba.lcut(text)
        reserve_set = list(set(token_result) - stop_words)
        result = [token for token in token_result if token in reserve_set]
        return result

    def similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings."""
        if not s1 or not s2:
            return 0.0
        s1 = s1.lower()
        s2 = s2.lower()
        if self.lang not in {"zh", "zh-ch", "zh-tw"}:
            punctuation_regex = re.compile(r"\p{P}|[ ]")
            s1_set = set(punctuation_regex.split(s1))
            s2_set = set(punctuation_regex.split(s2))
        else:
            s1_set = set(self.zh_tokenize(s1))
            s2_set = set(self.zh_tokenize(s2))
        intersection = s1_set.intersection(s2_set)
        union = s1_set.union(s2_set)
        if len(union) == 0 or len(s1_set) == 0:
            return 0.0
        return len(intersection) / math.log2(len(s1_set) + 1)

    def title_extractor(self, line: str):
        """Extract title and candidates from HTML."""
        try:
            element = fromstring(line)
        except Exception:
            return ""

        etree.strip_elements(element, *self.CONTENT_EXTRACTOR_USELESS_TAGS)
        etree.strip_tags(element, *self.CONTENT_EXTRACTOR_STRIP_TAGS)
        self.remove_children(element, self.CONTENT_EXTRACTOR_NOISE_XPATHS)

        title_extracted_by_meta = self.extract_by_meta(element)
        title_extracted_by_h = self.extract_by_h(element)
        title_extracted_by_hs = self.extract_by_hs(element)
        title_extracted_by_title = self.extract_by_title(element)

        title_final = ""
        is_pending = True
        if title_extracted_by_meta:
            title_final = title_extracted_by_meta
            is_pending = False

        if is_pending:
            v2_result = self.html_extractor_v2(line, 15)
            main_text = v2_result[0] if isinstance(v2_result, tuple) else ""
            title_extracted_by_hs = sorted(
                title_extracted_by_hs,
                key=lambda x: self.similarity(
                    x.rstrip().lstrip(), title_extracted_by_title + "\n" + main_text
                ),
                reverse=True,
            )
            if title_extracted_by_hs:
                title_final = title_extracted_by_hs[0].rstrip().lstrip()
                is_pending = False

        if title_extracted_by_title and is_pending:
            title_final = title_extracted_by_title
            is_pending = False

        if is_pending:
            title_final = title_extracted_by_h or ""

        return (
            title_final,
            title_extracted_by_meta,
            title_extracted_by_h,
            title_extracted_by_hs,
            title_extracted_by_title,
        )

    def html_extractor_v3(self, line: str, score: int, subtitle_score: float):
        """Extract main text with title/subtitle detection (V3)."""
        if not line:
            return "", "", ""
        line = line.replace("&nbsp;", "")
        if line.lstrip() and line.lstrip()[0] != "<":
            line = line[line.find("<") :]
        simhash = ComputeSimhash()
        try:
            element = fromstring(line)
        except Exception:
            return "", "", ""

        etree.strip_elements(element, *self.CONTENT_EXTRACTOR_USELESS_TAGS)
        etree.strip_tags(element, *self.CONTENT_EXTRACTOR_STRIP_TAGS)
        self.remove_children(element, self.CONTENT_EXTRACTOR_NOISE_XPATHS)

        elements = element.xpath("//body")
        if not elements:
            return "", "", ""

        descendants_list = list(self.descendants(elements[0], True))
        for descendant in descendants_list:
            if (descendant.text and descendant.text.rstrip() != "") or (
                descendant.tail and descendant.tail.rstrip() != ""
            ):
                self.rebuild_tree(descendant)

        elements = element.xpath("//body")
        descendants_list = list(self.descendants(elements[0], True))
        text_dict: dict[str, dict] = {}
        for descendant in descendants_list:
            if (descendant.text and descendant.text.rstrip() != "") or (
                descendant.tail and descendant.tail.rstrip() != ""
            ):
                tagpath = self.get_tagpath(descendant).replace("|", "")
                css_class = "".join(descendant.xpath("./@class")).lower().replace("|", "")
                css_style = "".join(descendant.xpath("./@style")).lower().replace("|", "")
                type_key = tagpath + "|" + css_class + "|" + css_style
                text = descendant.text.rstrip().lstrip() if descendant.text else ""
                tail = descendant.tail.rstrip().lstrip() if descendant.tail else ""
                is_new = True
                type_key_hash = simhash.evaluate(type_key)
                for key in text_dict:
                    key_hash = simhash.evaluate(key)
                    if simhash.distance(type_key_hash, key_hash) <= score:
                        text_dict[key]["text"] = "\n".join([
                            text_dict[key]["text"],
                            text + tail,
                        ])
                        is_new = False
                        if descendant.tag == "a":
                            text_dict[key]["atext_len"] += len(
                                (text + tail)
                                .replace(" ", "")
                                .replace("\t", "")
                                .replace("\n", "")
                            )
                if is_new:
                    text_dict[type_key] = {
                        "text": text + tail,
                        "atext_len": len(
                            (text + tail)
                            .replace(" ", "")
                            .replace("\t", "")
                            .replace("\n", "")
                        )
                        if descendant.tag == "a"
                        else 0,
                    }

        max_length = 0
        max_key = ""
        for type_key in text_dict:
            text_value = text_dict[type_key]["text"]
            text_len = len(text_value.replace(" ", "").replace("\t", "").replace("\n", ""))
            if (
                max_length < text_len
                and text_dict[type_key]["atext_len"] / max(1, text_len) < 0.5
            ):
                max_length = text_len
                max_key = type_key

        if not max_key:
            return "", "", ""

        try:
            self.lang = langdetect.detect(text_dict[max_key]["text"])
        except Exception:
            self.lang = ""
        title_extracted_by_meta = self.extract_by_meta(element)
        title_extracted_by_hs: list[str] = []
        title_extracted_by_title = self.extract_by_title(element)
        max_key_hash = simhash.evaluate(max_key)
        for descendant in descendants_list:
            if (descendant.text and descendant.text.rstrip() != "") or (
                descendant.tail and descendant.tail.rstrip() != ""
            ):
                tagpath = self.get_tagpath(descendant).replace("|", "")
                css_class = "".join(descendant.xpath("./@class")).lower().replace("|", "")
                css_style = "".join(descendant.xpath("./@style")).lower().replace("|", "")
                type_key = tagpath + "|" + css_class + "|" + css_style
                type_key_hash = simhash.evaluate(type_key)
                if simhash.distance(type_key_hash, max_key_hash) <= score:
                    break
            if descendant.tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                title_extracted_by_hs.append(" ".join(descendant.xpath(".//text()")))

        main_title = ""
        is_pending = True
        if title_extracted_by_meta:
            main_title = title_extracted_by_meta.rstrip().lstrip()
            is_pending = False
        if is_pending:
            title_extracted_by_hs = sorted(
                title_extracted_by_hs,
                key=lambda x: self.similarity(
                    x.rstrip().lstrip(),
                    title_extracted_by_title + "\n" + text_dict[max_key]["text"],
                ),
                reverse=True,
            )
            if title_extracted_by_hs:
                main_title = title_extracted_by_hs[0].rstrip().lstrip()
                is_pending = False
        if title_extracted_by_title and is_pending:
            main_title = title_extracted_by_title.rstrip().lstrip()
            is_pending = False

        is_active = False
        sub_titles = []
        tmp_titles = []
        main_text = ""
        sub_text = ""
        for descendant in descendants_list:
            if (descendant.text and descendant.text.rstrip() != "") or (
                descendant.tail and descendant.tail.rstrip() != ""
            ):
                tagpath = self.get_tagpath(descendant).replace("|", "")
                css_class = "".join(descendant.xpath("./@class")).lower().replace("|", "")
                css_style = "".join(descendant.xpath("./@style")).lower().replace("|", "")
                type_key = tagpath + "|" + css_class + "|" + css_style
                type_key_hash = simhash.evaluate(type_key)
                if simhash.distance(type_key_hash, max_key_hash) <= score:
                    if is_active:
                        sub_titles.extend(tmp_titles)
                        if sub_text != "":
                            if main_text == "":
                                main_text = sub_text
                            else:
                                main_text = "\n".join([main_text, sub_text])
                    is_active = True
                    tmp_titles = []
                    text = descendant.text.rstrip().lstrip() if descendant.text else ""
                    tail = descendant.tail.rstrip().lstrip() if descendant.tail else ""
                    if text + tail != "":
                        if main_text != "":
                            main_text = "\n".join([main_text, text + tail])
                        else:
                            main_text = text + tail
                        sub_text = ""
                elif descendant.tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    sub_title_text = " ".join(descendant.xpath(".//text()")).rstrip().lstrip()
                    if sub_title_text == main_title:
                        is_active = True
                        tmp_titles = []
                    tmp_titles.append(
                        [
                            sub_title_text,
                            self.similarity(
                                sub_title_text.rstrip().lstrip(),
                                main_title + "\n" + text_dict[max_key]["text"],
                            ),
                        ]
                    )
                    if (
                        is_active
                        and self.similarity(
                            sub_title_text, main_title + "\n" + text_dict[max_key]["text"]
                        )
                        >= subtitle_score
                    ):
                        if sub_text == "":
                            sub_text = sub_title_text
                        else:
                            sub_text = "\n".join([sub_text, sub_title_text])

        return self.clean_main_text(main_text), main_title, sub_titles

    def switch_list(self, a: list, b: list):
        """Swap two lists."""
        return b.copy(), a.copy()

    def if_interwoven(
        self, max_key: str, type_key: str, text_dict: dict, interweave_time: int = 2
    ) -> bool:
        """Check if two text blocks interleave within a threshold."""
        max_index_list = text_dict[max_key]["index"].copy()
        type_index_list = text_dict[type_key]["index"].copy()
        if max_index_list[0] < type_index_list[0]:
            now = max_index_list
            another = type_index_list
        else:
            now = type_index_list
            another = max_index_list
        now.pop(0)
        while len(now) > 0 and len(another) > 0:
            if now[0] > another[0]:
                now, another = self.switch_list(now, another)
                interweave_time -= 1
                if interweave_time <= 0:
                    return True
            now.pop(0)
        return False

    def get_main_text_keys(
        self,
        max_key: str,
        text_dict: dict,
        threshold: float = V4_DEFAULT_MAIN_THRESHOLD,
        main_len_floor: int = V4_DEFAULT_MAIN_LEN_FLOOR,
        link_ratio_max: float = V4_DEFAULT_LINK_RATIO_MAX,
        avg_text_len_min: float = V4_DEFAULT_AVG_TEXT_LEN_MIN,
        interweave_time: int = V4_DEFAULT_INTERWEAVE_TIME,
    ):
        """Return keys representing main text blocks."""
        main_text_keys = set()
        for type_key in text_dict:
            type_key_trimmed = self.trim(text_dict[type_key]["text"]) or ""
            type_key_trimmed_len = len(type_key_trimmed)
            if type_key_trimmed_len == 0:
                continue
            if (
                type_key_trimmed_len
                >= max(main_len_floor, threshold * len(self.trim(text_dict[max_key]["text"]) or ""))
                and text_dict[type_key]["atext_len"]
                / type_key_trimmed_len
                < link_ratio_max
                and self.get_avg_text_len(text_dict[type_key]["text"]) >= avg_text_len_min
            ):
                if self.if_interwoven(
                    max_key,
                    type_key,
                    text_dict,
                    interweave_time=interweave_time,
                ):
                    main_text_keys.add(type_key)
        main_text_keys.add(max_key)
        return main_text_keys

    def is_main_text_key(
        self,
        type_key: str,
        score: int,
        main_text_keys: Optional[set[str]] = None,
    ) -> bool:
        """Check if a type key belongs to the main text cluster."""
        if main_text_keys is not None:
            simhash = ComputeSimhash()
            type_key_hash = simhash.evaluate(type_key)
            for main_key in main_text_keys:
                main_key_hash = simhash.evaluate(main_key)
                if simhash.distance(type_key_hash, main_key_hash) <= score:
                    return True
            return False
        self.simhash = self.simhash or ComputeSimhash()
        return bool(self.simhash.is_conflict_noadd(type_key, score))

    def get_tag_key(self, element: HtmlElement) -> str:
        """Generate the structural key for an element."""
        tagpath = self.get_tagpath(element).replace("|", "")
        css_class = "".join(element.xpath("./@class")).lower().replace("|", "")
        css_style = "".join(element.xpath("./@style")).lower().replace("|", "")
        tag_key = tagpath + "|" + css_class + "|" + css_style
        return tag_key

    def get_avg_text_len(self, text: str) -> float:
        """Return average length of non-empty lines."""
        length_s = []
        for sub_text in text.split("\n"):
            trimmed = self.trim(sub_text)
            if trimmed and len(trimmed) > 0:
                length_s.append(len(trimmed))
        return float(np.mean(length_s)) if length_s else 0.0

    def html_extractor_v4(
        self,
        line: str,
        score: int,
        subtitle_score: float,
        avg_text_len_min: float = V4_DEFAULT_AVG_TEXT_LEN_MIN,
        main_threshold: float = V4_DEFAULT_MAIN_THRESHOLD,
        main_len_floor: int = V4_DEFAULT_MAIN_LEN_FLOOR,
        link_ratio_max: float = V4_DEFAULT_LINK_RATIO_MAX,
        interweave_time: int = V4_DEFAULT_INTERWEAVE_TIME,
        footer_penalty_ratio: float = V4_FOOTER_PENALTY_RATIO,
    ):
        """Extract main text with interwoven block handling (V4)."""
        line = self.normalize_html_input(line)
        if not line:
            return "", "", ""
        self.simhash = ComputeSimhash()
        simhash = ComputeSimhash()
        try:
            element = fromstring(line)
        except Exception:
            return "", "", ""

        etree.strip_elements(element, *self.CONTENT_EXTRACTOR_USELESS_TAGS)
        etree.strip_tags(element, *self.CONTENT_EXTRACTOR_STRIP_TAGS)
        self.remove_children(element, self.CONTENT_EXTRACTOR_NOISE_XPATHS)

        elements = element.xpath("//body")
        if not elements:
            return "", "", ""

        descendants_list = list(self.descendants(elements[0], True))
        for descendant in descendants_list:
            if (descendant.text and descendant.text.strip() != "") or (
                descendant.tail and descendant.tail.strip() != ""
            ):
                self.rebuild_tree(descendant)

        elements = element.xpath("//body")
        descendants_list = list(self.descendants(elements[0], True))
        text_dict: dict[str, dict] = {}
        text_index = 0
        for descendant in descendants_list:
            if (descendant.text and descendant.text.strip() != "") or (
                descendant.tail and descendant.tail.strip() != ""
            ):
                type_key = self.get_tag_key(descendant)
                text = descendant.text.strip() if descendant.text else ""
                tail = descendant.tail.strip() if descendant.tail else ""
                merged_text = text + tail
                is_new = True
                type_key_hash = simhash.evaluate(type_key)
                for key in text_dict:
                    key_hash = simhash.evaluate(key)
                    if simhash.distance(type_key_hash, key_hash) <= score:
                        text_dict[key]["text"] = "\n".join([
                            text_dict[key]["text"],
                            merged_text,
                        ])
                        text_dict[key]["index"].append(text_index)
                        if descendant.tag == "a":
                            text_dict[key]["atext_len"] += len(self.trim(merged_text) or "")
                        is_new = False
                if is_new:
                    text_dict[type_key] = {
                        "text": merged_text,
                        "index": [text_index],
                        "atext_len": len(self.trim(merged_text) or "")
                        if descendant.tag == "a"
                        else 0,
                    }
                text_index += 1

        max_length = 0
        max_key = ""
        for type_key in text_dict:
            trimmed = self.trim(text_dict[type_key]["text"]) or ""
            if not trimmed:
                continue
            effective_len = len(trimmed)
            if self.has_footer_noise_keywords(text_dict[type_key]["text"]):
                effective_len = int(effective_len * footer_penalty_ratio)
            if (
                max_length < effective_len
                and text_dict[type_key]["atext_len"] / max(1, len(trimmed)) < link_ratio_max
                and self.get_avg_text_len(text_dict[type_key]["text"]) >= avg_text_len_min
            ):
                max_length = effective_len
                max_key = type_key

        if not max_key:
            return "", "", ""

        try:
            self.lang = langdetect.detect(text_dict[max_key]["text"])
        except Exception:
            self.lang = ""
        main_text_keys = self.get_main_text_keys(
            max_key,
            text_dict,
            threshold=main_threshold,
            main_len_floor=main_len_floor,
            link_ratio_max=link_ratio_max,
            avg_text_len_min=avg_text_len_min,
            interweave_time=interweave_time,
        )
        text_only = ""
        for descendant in descendants_list:
            if (descendant.text and descendant.text.strip() != "") or (
                descendant.tail and descendant.tail.strip() != ""
            ):
                type_key = self.get_tag_key(descendant)
                if self.is_main_text_key(type_key, score, main_text_keys):
                    text = descendant.text.strip() if descendant.text else ""
                    tail = descendant.tail.strip() if descendant.tail else ""
                    text_only += text + tail + "\n"

        title_keys = set()
        for descendant in descendants_list:
            if (
                descendant.tag in ["h1", "h2", "h3", "h4", "h5", "h6"]
                and self.trim("".join(descendant.xpath(".//text()"))) != ""
            ):
                title_key = self.get_tag_key(descendant)
                if (
                    title_key not in title_keys
                    and self.similarity("".join(descendant.xpath(".//text()")).strip(), text_only)
                    >= subtitle_score
                ):
                    title_keys.add(title_key)

        main_text = ""
        titles = []
        to_be_add = ""
        for descendant in descendants_list:
            if (
                descendant.tag in ["h1", "h2", "h3", "h4", "h5", "h6"]
                and self.trim("".join(descendant.xpath(".//text()"))) != ""
            ):
                title_key = self.get_tag_key(descendant)
                title_text = "".join(descendant.xpath(".//text()"))
                if title_key in title_keys:
                    to_be_add += title_text + "\n"
                    titles.append([self.similarity(title_text.strip(), text_only), title_key, title_text])
                else:
                    titles.append([self.similarity(title_text.strip(), text_only), title_key, title_text])
            if (descendant.text and descendant.text.strip() != "") or (
                descendant.tail and descendant.tail.strip() != ""
            ):
                type_key = self.get_tag_key(descendant)
                if self.is_main_text_key(type_key, score, main_text_keys):
                    text = descendant.text.strip() if descendant.text else ""
                    tail = descendant.tail.strip() if descendant.tail else ""
                    if to_be_add != "":
                        main_text += to_be_add + "\n"
                        to_be_add = ""
                    main_text += text + tail + "\n"

        return self.clean_main_text(main_text), "", titles

    # Backwards-compatible camelCase aliases
    def htmlExtractor(self, line: str) -> str:  # noqa: N802
        return self.html_extractor(line)

    def htmlExtractor_V2(self, line: str, score: int):  # noqa: N802
        return self.html_extractor_v2(line, score)

    def htmlExtractor_V3(self, line: str, score: int, subtitle_score: float):  # noqa: N802
        return self.html_extractor_v3(line, score, subtitle_score)

    def htmlExtractor_V4(self, line: str, score: int, subtitle_score: float):  # noqa: N802
        return self.html_extractor_v4(line, score, subtitle_score)

    def titleExtractor(self, line: str):  # noqa: N802
        return self.title_extractor(line)

    # Higher-level convenience APIs
    def extract_main_text(
        self,
        html: str,
        score: int = V4_DEFAULT_SCORE,
        subtitle_score: float = V4_DEFAULT_SUBTITLE_SCORE,
    ) -> str:
        """Return only main text using V4."""
        result = self.html_extractor_v4(html, score, subtitle_score)
        if not isinstance(result, tuple) or len(result) < 1:
            return ""
        main_text = result[0]
        if not isinstance(main_text, str):
            return ""
        return main_text

    def extract(
        self,
        html: str,
        score: int = V4_DEFAULT_SCORE,
        subtitle_score: float = V4_DEFAULT_SUBTITLE_SCORE,
    ):
        """Return main text, title, and title candidates."""
        v4_result = self.html_extractor_v4(html, score, subtitle_score)
        if isinstance(v4_result, tuple) and len(v4_result) == 3:
            main_text, _, titles = v4_result
        else:
            main_text, titles = "", []
        if not isinstance(main_text, str):
            main_text = ""
        if not isinstance(titles, list):
            titles = []
        title_result = self.title_extractor(html)
        title = title_result[0] if isinstance(title_result, tuple) else ""
        return main_text, title, titles

    def clean_mainText(self, main_text: str) -> str:  # noqa: N802
        return self.clean_main_text(main_text)

    def zhTokenlize(self, text: str) -> List[str]:  # noqa: N802
        return self.zh_tokenize(text)

    def get_mainText_keys(self, max_key: str, text_dict: dict, threshold: float = 0.7):  # noqa: N802
        return self.get_main_text_keys(max_key, text_dict, threshold=threshold)

    def is_mainText_key(  # noqa: N802
        self,
        type_key: str,
        score: int,
        main_text_keys: Optional[set[str]] = None,
    ) -> bool:
        return self.is_main_text_key(type_key, score, main_text_keys)

    def get_tagKey(self, element: HtmlElement) -> str:  # noqa: N802
        return self.get_tag_key(element)

    def get_avgTextLen(self, text: str) -> float:  # noqa: N802
        return self.get_avg_text_len(text)
