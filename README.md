# PDF Translation Pipeline

A high-quality PDF translation system that preserves document layout while translating academic papers from English to Chinese.

## Features

- **Layout-Preserving Translation**: Maintains original PDF formatting, fonts, and visual structure
- **Fixed-Length Translation (Iso-Length)**: Intelligent character count optimization to fit translated text within original bounding boxes
- **Formula Rendering**: High-quality LaTeX formula rendering with configurable DPI
- **Multi-round Reflow**: Iterative text adjustment to meet layout requirements
- **Configurable Pipeline**: YAML-based configuration for API settings, processing options, and rendering parameters

## Project Structure

```
agentParse/
├── config.yaml              # Main configuration file (API settings)
├── run_translation.py       # Main translation pipeline (full batch)
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── data-new/                # Full batch input data
│   ├── general_papers
│   ├── formula_papers
│   └── figure_table_papers
├── runs/                    # Runtime outputs
│   └── output/              # Translated PDF outputs
│
├── ablation/               # Ablation experiment scripts
│   ├── run_isolength_ablation.py     # Iso-length + multi-round reflow
│   ├── run_no_isolength_ablation.py  # No iso-length, no reflow
│   └── README.md                     # Ablation experiment documentation
│
├── ablation_data/          # Ablation experiment data (separate)
│   └── input/              # Ablation input papers
│
├── scripts/                # Core translation components
│   ├── TranslationAgent.py           # Translation with iso-length
│   ├── TranslationAgentNoIsoLength.py # Translation without iso-length
│   ├── ReflowAgent.py                # Layout reflow agent
│   └── ReflowTask.py                 # Reflow task utilities
│
├── tools/                  # Low-level tools
│   ├── api_client.py        # Generic API client
│   ├── pdf_reflow_tool.py   # PDF manipulation and rendering
│   └── text_refiner.py      # Text processing utilities
│
├── fonts/                   # Chinese and English fonts
└── logs/                    # Runtime logs
```

## Installation

### Requirements

- Python 3.8+
- PyMuPDF (fitz)
- PIL/Pillow
- matplotlib
- PyYAML
- requests

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API settings in `config.yaml`:
```yaml
translation:
  api:
    base_url: "https://your-api-provider.com/v1"
    api_key: "your-api-key"
    model: "your-model-name"
```

   Or use environment variables:
```yaml
translation:
  api:
    api_key: "${MY_API_KEY}"
```

3. Prepare input data:
   - Place each paper under `data-new/{category}/{layout}/{paper_id}/hybrid_auto/`
   - The pipeline prefers:
     - `{paper_id}_enriched.json` - Enriched content extraction data
     - `{paper_id}_origin.pdf` - Original PDF
   - It also keeps backward compatibility with the old `data/input/.../{paper_id}_content_list_customize.json` layout
   - Recommended category names: `general_papers`, `formula_papers`, `figure_table_papers`
   - Recommended layout names: `single_column`, `double_column`, `complex_layout`
   - CLI/config aliases `formula/figure/general` plus the old Chinese names are still supported

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
# Translation settings
translation:
  api:
    base_url: ""                  # Your API endpoint (e.g., https://api.openai.com/v1)
    api_key: ""                   # Your API key
    model: ""                     # Model name (e.g., gpt-4o, claude-3-opus)
  enable_isolength: true          # Enable fixed-length translation
  target_language: "zh"           # Target language
  concurrency: 4

# Reflow settings (can use same or different API)
reflow:
  api:
    base_url: ""                  # Your API endpoint
    api_key: ""                   # Your API key
    model: ""                     # Model name
  enabled: true
  max_rounds: 3                   # Maximum total rounds, including M0

# Rendering settings
rendering:
  formula_dpi: 300                # Formula rendering clarity
```

**Note:** All API settings must be configured before running. The pipeline supports any OpenAI-compatible API.

## Usage

### Basic Usage

```bash
# Run with default configuration
python run_translation.py

# Use custom config file
python run_translation.py --config my_config.yaml

# Process specific category
python run_translation.py --category formula

# Process specific layout
python run_translation.py --layout single_column

# Parallel processing
python run_translation.py --workers 4
```

### Input Data Structure

```
data-new/
├── formula_papers/
│   ├── single_column/
│   │   └── 2512.12345/
│   │       └── hybrid_auto/
│   │           ├── 2512.12345_origin.pdf
│   │           └── 2512.12345_enriched.json
│   ├── double_column/
│   └── complex_layout/
├── figure_table_papers/
│   └── ...
└── general_papers/
    └── ...
```

### Output Structure

```
runs/output/
├── formula_papers/
│   └── single_column/
│       └── 2512.12345/
│           ├── 2512.12345_translated.pdf
│           └── 2512.12345_translated.json
└── translation_results.json
```

## Pipeline Stages

1. **Content Extraction**: Load pre-extracted content blocks from JSON
2. **Rich Text Extraction**: Extract font styles from original PDF
3. **Translation**: Translate text with optional iso-length optimization
4. **Reflow**: Iteratively adjust text to fit layout constraints
5. **Rendering**: Generate final PDF with translated content

## Key Components

### TranslationAgent
- Handles text translation with fixed-length optimization
- Calculates target character counts based on bounding box size
- Supports formula masking during translation

### ReflowAgent
- Manages layout reflow process
- Adjusts font size and spacing to fit text
- Supports multi-round optimization

### PDFReflowTool
- Low-level PDF manipulation
- Formula rendering with LaTeX
- Text and image placement

## Ablation Experiments

Ablation experiments are in the `ablation/` directory:

```bash
# Run iso-length experiment (saves results at each reflow round)
python ablation/run_isolength_ablation.py --max-rounds 3

# Run no-isolength experiment (control group, no reflow)
python ablation/run_no_isolength_ablation.py
```

**Experiments:**
- `run_isolength_ablation.py` - Iso-length translation with multi-round reflow
  - Saves JSON + PDF for each round (round_0, round_1, round_2, round_3)
  - Output: `ablation_data/output/`
  
- `run_no_isolength_ablation.py` - No iso-length, forced no reflow
  - Control group for comparison
  - Output: `ablation_data/output_no_isolength/`

See `ablation/README.md` for detailed documentation.

## API Configuration

The pipeline supports any OpenAI-compatible API provider. Configure your API endpoints in `config.yaml`:

```yaml
translation:
  api:
    base_url: "https://your-api-provider.com/v1"  # API endpoint
    api_key: "sk-xxxxx"                            # API key
    model: "your-model"                            # Model name

reflow:
  api:
    base_url: "https://your-api-provider.com/v1"  # Can be same or different
    api_key: "sk-xxxxx"
    model: "your-model"
```

You can also use environment variables for API keys:
```yaml
api_key: "${MY_API_KEY}"  # Will expand from environment
```

## Troubleshooting

### Formula Rendering Issues
- Ensure matplotlib and LaTeX fonts are installed
- Increase `rendering.formula_dpi` for clearer formulas

### API Rate Limits
- Reduce `concurrency` in config
- API clients include built-in retry logic

### Missing Fonts
- Place required fonts in `fonts/` directory
- Default Chinese font: Microsoft YaHei (MSYH.TTF)

## License

Internal use only.

## Contact

For questions or issues, contact the development team.
