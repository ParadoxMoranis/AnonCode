#!/usr/bin/env python3
"""
Analyze page-level text coverage for translated PDFs.

The script compares per-page expected translated text volume from
`*_translated.json` with actual extractable text volume from the final
`*_translated.pdf`, then emits a Markdown report with per-page risk labels.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import fitz


NON_TEXT_TYPES = {
    "image",
    "figure",
    "table",
    "equation",
    "formula",
    "line",
    "rect",
    "curve",
}


def normalize_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", "", s)
    return s


def translated_char_count(item: dict) -> int:
    item_type = str(item.get("type") or "").lower()
    if item_type in NON_TEXT_TYPES:
        return 0
    translated = normalize_text(str(item.get("translated") or ""))
    return len(translated)


@dataclass
class PageStats:
    page_no: int
    expected_chars: int = 0
    actual_chars: int = 0
    total_items: int = 0
    text_items: int = 0
    formula_items: int = 0
    reference_items: int = 0
    title_items: int = 0
    header_footer_items: int = 0
    non_text_items: int = 0

    @property
    def ratio(self) -> float:
        if self.expected_chars <= 0:
            return 1.0 if self.actual_chars > 0 else 0.0
        return self.actual_chars / self.expected_chars

    def label(self) -> str:
        if self.expected_chars <= 0:
            return "非主文本页"
        formula_dense = self.formula_items >= 4 or (
            self.total_items > 0 and self.formula_items / self.total_items >= 0.35
        )
        if self.ratio >= 0.82:
            return "OK"
        if formula_dense and self.ratio >= 0.5:
            return "公式页低抽取"
        if self.reference_items >= 3 and self.ratio >= 0.55:
            return "参考文献页低抽取"
        return "建议视觉复核"


def iter_pdf_paths(root: Path):
    for pdf_path in sorted(root.rglob("*_translated.pdf")):
        json_path = pdf_path.with_name(pdf_path.name.replace("_translated.pdf", "_translated.json"))
        if json_path.exists():
            yield pdf_path, json_path


def collect_page_stats(pdf_path: Path, json_path: Path) -> list[PageStats]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    by_page: dict[int, PageStats] = defaultdict(lambda: PageStats(page_no=0))

    for item in data:
        page_idx = item.get("page_idx")
        if page_idx is None:
            continue
        page_no = int(page_idx) + 1
        stats = by_page.setdefault(page_no, PageStats(page_no=page_no))
        stats.page_no = page_no
        stats.total_items += 1

        item_type = str(item.get("type") or "").lower()
        detected_type = str(item.get("detected_type") or "").lower()

        if item_type in NON_TEXT_TYPES:
            stats.non_text_items += 1
        else:
            stats.text_items += 1
            stats.expected_chars += translated_char_count(item)

        if item_type in {"equation", "formula"} or detected_type in {"equation", "formula"}:
            stats.formula_items += 1
        if item_type == "reference" or detected_type == "reference":
            stats.reference_items += 1
        if detected_type in {"title", "section_header"}:
            stats.title_items += 1
        if item_type in {"header", "page_footnote", "page_number"} or detected_type in {
            "footnote",
            "footer",
        }:
            stats.header_footer_items += 1

    doc = fitz.open(pdf_path)
    for page_index in range(doc.page_count):
        page_no = page_index + 1
        stats = by_page.setdefault(page_no, PageStats(page_no=page_no))
        text = normalize_text(doc[page_index].get_text("text"))
        stats.actual_chars = len(text)

    return [by_page[i] for i in range(1, doc.page_count + 1)]


def paper_title(pdf_path: Path, root: Path) -> str:
    rel = pdf_path.relative_to(root)
    return str(rel.parent)


def render_markdown(root: Path, output_path: Path) -> dict:
    lines: list[str] = []
    totals = Counter()
    suspicious: list[tuple[str, int, float, int, int, str]] = []

    lines.append("# 9类测试逐页核查报告")
    lines.append("")
    lines.append(f"- 运行目录: `{root}`")
    lines.append("- 统计口径: `expected_chars` 来自 `*_translated.json` 的逐页文本块译文长度，`actual_chars` 来自最终 PDF 的逐页可抽取文本长度。")
    lines.append("- 风险标签: `OK` / `公式页低抽取` / `参考文献页低抽取` / `建议视觉复核` / `非主文本页`。")
    lines.append("")

    for pdf_path, json_path in iter_pdf_paths(root):
        title = paper_title(pdf_path, root)
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| 页码 | expected_chars | actual_chars | ratio | blocks | formula | refs | label |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")

        page_stats = collect_page_stats(pdf_path, json_path)
        paper_counter = Counter()
        for stats in page_stats:
            label = stats.label()
            paper_counter[label] += 1
            totals[label] += 1
            if label == "建议视觉复核":
                suspicious.append(
                    (
                        title,
                        stats.page_no,
                        stats.ratio,
                        stats.expected_chars,
                        stats.actual_chars,
                        label,
                    )
                )
            lines.append(
                f"| {stats.page_no} | {stats.expected_chars} | {stats.actual_chars} | {stats.ratio:.3f} | "
                f"{stats.total_items} | {stats.formula_items} | {stats.reference_items} | {label} |"
            )

        lines.append("")
        lines.append(
            "- 小结: "
            f"OK={paper_counter['OK']}, 公式页低抽取={paper_counter['公式页低抽取']}, "
            f"参考文献页低抽取={paper_counter['参考文献页低抽取']}, "
            f"建议视觉复核={paper_counter['建议视觉复核']}, 非主文本页={paper_counter['非主文本页']}"
        )
        lines.append("")

    suspicious.sort(key=lambda x: (x[2], -x[3]))

    lines.append("## 总体汇总")
    lines.append("")
    total_pages = sum(totals.values())
    lines.append(f"- 总页数: {total_pages}")
    for key in ["OK", "公式页低抽取", "参考文献页低抽取", "建议视觉复核", "非主文本页"]:
        lines.append(f"- {key}: {totals[key]}")
    lines.append("")

    lines.append("## 优先复核页")
    lines.append("")
    if suspicious:
        lines.append("| 论文 | 页码 | ratio | expected_chars | actual_chars |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for title, page_no, ratio, expected_chars, actual_chars, _ in suspicious[:30]:
            lines.append(
                f"| {title} | {page_no} | {ratio:.3f} | {expected_chars} | {actual_chars} |"
            )
    else:
        lines.append("- 无 `建议视觉复核` 页。")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "total_pages": total_pages,
        "totals": dict(totals),
        "suspicious": suspicious,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze translated PDF page coverage")
    parser.add_argument("root", help="Run directory root")
    parser.add_argument(
        "--output",
        help="Markdown output path",
        default="9类测试-逐页核查报告.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_path = Path(args.output).resolve()
    summary = render_markdown(root, output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
