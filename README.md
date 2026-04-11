# Anonymous PDF Translation Pipeline

This repository contains the core code for a layout-preserving academic PDF translation system.

## Included

- Main pipeline entry: `run_translation.py`
- Core agents: `scripts/`
- Rendering / API / refinement utilities: `tools/`
- Required fonts for PDF rendering: `fonts/`
- Lightweight experiment runners:
  - `run_model_robustness.py`
  - `run_operation_ablation.py`
  - `ablation/`
- Attribution analysis helper:
  - `analyze_planner_attribution.py`

## Excluded

The following are intentionally not included in this anonymous code release:

- Input datasets
- Runtime outputs and cached artifacts
- Logs and temporary files
- Old one-off scripts and environment-specific utilities

## Repository Structure

```text
Anonymous/
├── config.yaml
├── model_matrix.example.json
├── requirements.txt
├── run_translation.py
├── run_model_robustness.py
├── run_operation_ablation.py
├── analyze_planner_attribution.py
├── scripts/
├── tools/
├── fonts/
└── ablation/
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` before running:

```yaml
translation:
  api:
    base_url: "https://your-api-provider/v1"
    api_key: "${YOUR_TRANSLATION_API_KEY}"
    model: "your-translation-model"

reflow:
  api:
    base_url: "https://your-api-provider/v1"
    api_key: "${YOUR_REFLOW_API_KEY}"
    model: "your-reflow-model"
```

## Expected Input Layout

Input data is expected to follow:

```text
data-new/
├── 一般论文/
├── 图表论文/
└── 公式论文/
    └── {layout}/
        └── {paper_id}/
            └── hybrid_auto/
                ├── {paper_id}_origin.pdf
                └── {paper_id}_enriched.json
```

The repository does not ship with these data files.

## Usage

```bash
python run_translation.py
python run_translation.py --config config.yaml
python run_translation.py --category formula
python run_translation.py --layout single_column
python run_translation.py --workers 4
```

## Core Pipeline

1. Load parsed PDF content blocks.
2. Extract rich-text style cues from the original PDF.
3. Translate text with optional iso-length control.
4. Run iterative reflow for fit / overflow / underflow correction.
5. Render the translated PDF.

## Notes

- `config.yaml` is a template and does not contain usable API credentials.
- Data, outputs, and large experiment artifacts should remain outside this repository.
