import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Optional


class ParseAgent:
    """
    Lightweight parser orchestration layer.

    Responsibilities:
    1. Reuse already parsed MinerU / enriched outputs when present.
    2. Optionally invoke an external parser command when only PDF is available.
    3. Normalize discovered assets for the translation pipeline.
    """

    JSON_CANDIDATES = (
        "{doc_id}_enriched.json",
        "{doc_id}_content_list_customize.json",
        "{doc_id}_content_list.json",
        "{doc_id}_content_list_v2.json",
    )

    PDF_CANDIDATES = (
        "{doc_id}_origin.pdf",
        "{doc_id}.pdf",
    )

    def __init__(self, config: Optional[Dict] = None, base_dir: Optional[Path] = None):
        self.config = config or {}
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        parse_cfg = self.config.get("parse", {})
        self.enabled = parse_cfg.get("enabled", False)
        self.command_template = parse_cfg.get("command_template", "")
        self.output_subdir = parse_cfg.get("output_subdir", "hybrid_auto")
        self.prefer_enriched = parse_cfg.get("prefer_enriched", True)

    def ensure_document_assets(self, doc_dir: Path, doc_id: str) -> Optional[Dict]:
        assets = self.find_existing_assets(doc_dir, doc_id)
        if assets:
            return assets

        if not self.enabled:
            return None

        pdf_path = self._find_pdf_only(doc_dir, doc_id)
        if not pdf_path:
            return None

        self.run_external_parser(pdf_path, doc_dir, doc_id)
        return self.find_existing_assets(doc_dir, doc_id)

    def find_existing_assets(self, doc_dir: Path, doc_id: str) -> Optional[Dict]:
        candidate_roots = []
        parse_dir = doc_dir / self.output_subdir
        if parse_dir.exists():
            candidate_roots.append(parse_dir)
        candidate_roots.append(doc_dir)

        json_names = list(self.JSON_CANDIDATES)
        if self.prefer_enriched:
            json_names.sort(key=lambda name: 0 if "enriched" in name else 1)

        for root in candidate_roots:
            json_path = next(
                (root / name.format(doc_id=doc_id) for name in json_names if (root / name.format(doc_id=doc_id)).exists()),
                None,
            )
            pdf_path = next(
                (
                    root / name.format(doc_id=doc_id)
                    for name in self.PDF_CANDIDATES
                    if (root / name.format(doc_id=doc_id)).exists()
                ),
                None,
            )
            if json_path and pdf_path:
                return {
                    "input_dir": root,
                    "json_path": json_path,
                    "pdf_path": pdf_path,
                    "source": "existing",
                }
        return None

    def run_external_parser(self, pdf_path: Path, doc_dir: Path, doc_id: str) -> None:
        if not self.command_template:
            raise RuntimeError(
                "parse.enabled=true but parse.command_template is empty; "
                "configure an external parser command first."
            )

        output_dir = doc_dir / self.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        replacements = {
            "pdf_path": str(pdf_path),
            "doc_dir": str(doc_dir),
            "output_dir": str(output_dir),
            "doc_id": doc_id,
        }
        command = self.command_template.format(**replacements)
        completed = subprocess.run(
            shlex.split(command),
            cwd=str(self.base_dir),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Parser command failed ({completed.returncode}): {completed.stderr.strip() or completed.stdout.strip()}"
            )

    def export_parse_manifest(self, doc_dir: Path, doc_id: str, output_path: Path) -> None:
        assets = self.find_existing_assets(doc_dir, doc_id)
        manifest = {
            "doc_id": doc_id,
            "doc_dir": str(doc_dir),
            "parse_enabled": self.enabled,
            "assets": {
                "pdf_path": str(assets["pdf_path"]) if assets else None,
                "json_path": str(assets["json_path"]) if assets else None,
                "input_dir": str(assets["input_dir"]) if assets else None,
            },
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def _find_pdf_only(self, doc_dir: Path, doc_id: str) -> Optional[Path]:
        for name in self.PDF_CANDIDATES:
            candidate = doc_dir / name.format(doc_id=doc_id)
            if candidate.exists():
                return candidate
        for candidate in doc_dir.glob("*.pdf"):
            return candidate
        return None
