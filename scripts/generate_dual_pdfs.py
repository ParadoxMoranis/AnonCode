#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import fitz


def find_origin_pdf(doc_dir: Path, doc_id: str) -> Path:
    candidates = sorted(doc_dir.glob("hybrid_auto/*_origin.pdf"))
    if not candidates:
        raise FileNotFoundError(f"Missing origin PDF under {doc_dir / 'hybrid_auto'}")
    return candidates[0]


def make_dual_pdf(origin_pdf: Path, translated_pdf: Path, output_pdf: Path) -> None:
    origin_doc = fitz.open(str(origin_pdf))
    translated_doc = fitz.open(str(translated_pdf))
    out_doc = fitz.open()

    try:
        page_count = max(origin_doc.page_count, translated_doc.page_count)
        for page_idx in range(page_count):
            origin_page = origin_doc[page_idx] if page_idx < origin_doc.page_count else None
            translated_page = translated_doc[page_idx] if page_idx < translated_doc.page_count else None

            origin_rect = origin_page.rect if origin_page else fitz.Rect(0, 0, 595, 842)
            translated_rect = translated_page.rect if translated_page else fitz.Rect(0, 0, 595, 842)

            out_width = origin_rect.width + translated_rect.width
            out_height = max(origin_rect.height, translated_rect.height)
            out_page = out_doc.new_page(width=out_width, height=out_height)

            if origin_page:
                left_rect = fitz.Rect(
                    0,
                    0,
                    origin_rect.width,
                    origin_rect.height,
                )
                out_page.show_pdf_page(left_rect, origin_doc, page_idx)

            if translated_page:
                right_rect = fitz.Rect(
                    origin_rect.width,
                    0,
                    origin_rect.width + translated_rect.width,
                    translated_rect.height,
                )
                out_page.show_pdf_page(right_rect, translated_doc, page_idx)

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        out_doc.save(str(output_pdf), garbage=4, deflate=True)
    finally:
        out_doc.close()
        translated_doc.close()
        origin_doc.close()


def zip_dual_pdfs(zip_path: Path, dual_pdfs: list[Path], base_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdf_path in dual_pdfs:
            zf.write(pdf_path, pdf_path.relative_to(base_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate side-by-side origin/translated dual PDFs.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing *_translated.pdf outputs.")
    parser.add_argument("--data-root", required=True, help="Root directory of source papers, e.g. data-new/formula_papers.")
    parser.add_argument("--zip-path", required=True, help="Output zip path.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    data_root = Path(args.data_root).resolve()
    zip_path = Path(args.zip_path).resolve()

    translated_pdfs = sorted(run_dir.glob("**/*_translated.pdf"))
    if not translated_pdfs:
        raise FileNotFoundError(f"No translated PDFs found under {run_dir}")

    dual_pdfs: list[Path] = []
    for translated_pdf in translated_pdfs:
        doc_id = translated_pdf.stem.removesuffix("_translated")
        layout = translated_pdf.parent.parent.name
        source_doc_dir = data_root / layout / doc_id
        origin_pdf = find_origin_pdf(source_doc_dir, doc_id)
        dual_pdf = translated_pdf.parent / f"{doc_id}_dual.pdf"
        make_dual_pdf(origin_pdf, translated_pdf, dual_pdf)
        dual_pdfs.append(dual_pdf)
        print(f"[OK] {dual_pdf}")

    zip_dual_pdfs(zip_path, dual_pdfs, run_dir)
    print(f"[ZIP] {zip_path}")
    print(f"[COUNT] {len(dual_pdfs)} dual PDFs")


if __name__ == "__main__":
    main()
