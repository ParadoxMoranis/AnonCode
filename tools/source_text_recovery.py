import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import fitz


RECOVERABLE_TYPES = {
    "text",
    "title",
    "section_header",
    "page_header",
    "header",
    "footer",
    "caption",
    "figure_caption",
    "table_caption",
    "footnote",
    "abstract",
}

ENUM_MARKER_RE = re.compile(
    r"\(?"
    r"(?:"
    r"\d+"
    r"|[A-Za-z]"
    r"|[ivxlcdmIVXLCDM]+"
    r")"
    r"\)?[.)]?"
)
INFO_TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+|[=<>+\-*/\\|[\]{}()^_~:;,.'\"`$%&!?@#]+")


def _extract_enumeration_markers(text: str) -> List[str]:
    compact = _normalize_spaces(text)
    if not compact:
        return []

    markers = []
    cursor = 0
    for match in ENUM_MARKER_RE.finditer(compact):
        if match.start() != cursor:
            gap = compact[cursor:match.start()]
            if gap.strip():
                return []
        markers.append(match.group(0))
        cursor = match.end()

    if cursor != len(compact) and compact[cursor:].strip():
        return []
    return markers


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _strip_control_chars(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch >= " " or ch in "\n\t")


def _information_score(text: str) -> int:
    compact = _normalize_spaces(text)
    if not compact:
        return 0
    return sum(len(token) for token in INFO_TOKEN_RE.findall(compact))


def _looks_low_information(text: str) -> bool:
    compact = _normalize_spaces(text)
    if not compact:
        return True

    markers = _extract_enumeration_markers(compact)
    if len(markers) >= 2:
        return True

    non_marker = re.sub(r"[\s(){}\[\]\d.,;:<>+\-*/\\|]+", "", compact)
    return len(non_marker) <= 2 and compact.count("(") >= 1


def _clean_local_md(local_md: str) -> str:
    if not local_md:
        return ""
    text = _strip_control_chars(local_md).replace("**", "")
    text = text.replace("*", "")
    text = text.replace("•", "")
    return _normalize_spaces(text)


def _extract_clip_text(doc: fitz.Document, item: Dict) -> str:
    try:
        page_idx = int(item.get("page_idx", item.get("page", 0)))
        bbox = item.get("bbox") or []
        if len(bbox) != 4 or page_idx < 0 or page_idx >= len(doc):
            return ""
        page = doc[page_idx]
        rect = fitz.Rect(
            max(page.rect.x0, float(bbox[0]) - 0.5),
            max(page.rect.y0, float(bbox[1]) - 0.5),
            min(page.rect.x1, float(bbox[2]) + 0.5),
            min(page.rect.y1, float(bbox[3]) + 0.5),
        )
        return _normalize_spaces(_strip_control_chars(page.get_text("text", clip=rect)))
    except Exception:
        return ""


def _meaningfully_richer(original: str, candidate: str) -> bool:
    base = _normalize_spaces(original)
    cand = _normalize_spaces(candidate)
    if not cand or cand == base:
        return False

    base_score = _information_score(base)
    cand_score = _information_score(cand)
    if cand_score >= base_score + 16:
        return True
    return cand_score >= max(8, base_score * 2) and len(cand) >= len(base) + 10


def recover_low_information_items(data: List[Dict], doc: fitz.Document) -> Dict[str, object]:
    repaired = 0
    examples: List[Tuple[int, str, str]] = []

    for item in data:
        item_type = (item.get("type") or "").lower()
        if item_type not in RECOVERABLE_TYPES:
            continue

        original_text = item.get("text") or ""
        local_md = item.get("local_md") or ""
        if not _looks_low_information(original_text):
            continue

        best_source: Optional[str] = None
        best_candidate = ""

        pdf_candidate = _extract_clip_text(doc, item)
        if _meaningfully_richer(original_text, pdf_candidate):
            best_source = "pdf_text"
            best_candidate = pdf_candidate

        local_candidate = _clean_local_md(local_md)
        if _meaningfully_richer(original_text, local_candidate):
            if not best_candidate or _information_score(local_candidate) > _information_score(best_candidate):
                best_source = "local_md"
                best_candidate = local_candidate

        if not best_candidate:
            continue

        item["text"] = best_candidate
        item["_recovered_text_source"] = best_source
        repaired += 1
        if len(examples) < 5:
            page_no = int(item.get("page_idx", item.get("page", 0))) + 1
            examples.append((page_no, best_source or "unknown", best_candidate[:140]))

    refreshed_contexts = 0
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for item in data:
        lid = item.get("logical_para_id")
        if lid is None:
            continue
        groups[str(lid)].append(item)

    for items in groups.values():
        ordered = sorted(
            items,
            key=lambda entry: (
                int(entry.get("page_idx", entry.get("page", 0))),
                float((entry.get("bbox") or [0, 0, 0, 0])[1]),
                float((entry.get("bbox") or [0, 0, 0, 0])[0]),
            ),
        )
        rebuilt_context = " ".join(_normalize_spaces(item.get("text") or "") for item in ordered).strip()
        if not rebuilt_context:
            continue

        current_context = _normalize_spaces(ordered[0].get("context") or "")
        if current_context == rebuilt_context:
            continue
        if not current_context or _looks_low_information(current_context) or _meaningfully_richer(current_context, rebuilt_context):
            for item in ordered:
                item["context"] = rebuilt_context
            refreshed_contexts += 1

    return {
        "repaired": repaired,
        "refreshed_contexts": refreshed_contexts,
        "examples": examples,
    }
