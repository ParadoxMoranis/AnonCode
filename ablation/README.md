# Ablation Experiments

This directory contains ablation experiment scripts to evaluate the effectiveness of:
1. **Iso-Length Translation** (fixed-length optimization)
2. **Reflow Iterations** (multi-round layout adjustment)

## Experiment Scripts

### 1. `run_isolength_ablation.py` - Iso-Length with Multiple Reflow Rounds

Runs translation **with** fixed-length optimization and saves results at each stage.

**Output Structure:**
```
ablation_data/output/
└── {category}/
    └── {layout}/
        └── {paper_id}/
            ├── round_0/         # After translation, before reflow
            │   ├── {id}.json
            │   └── {id}.pdf
            ├── round_1/         # After 1st reflow
            │   └── ...
            ├── round_2/         # After 2nd reflow
            │   └── ...
            └── round_3/         # After 3rd reflow
                └── ...
```

**Usage:**
```bash
# Run with default 3 reflow rounds
python ablation/run_isolength_ablation.py

# Specify max rounds
python ablation/run_isolength_ablation.py --max-rounds 5

# Process specific category
python ablation/run_isolength_ablation.py --category Formula
```

### 2. `run_no_isolength_ablation.py` - No Iso-Length (Control Group)

Runs translation **without** fixed-length optimization.
**Forced: max_rounds = 0** (no reflow iterations)

This serves as a control group to demonstrate the effectiveness of iso-length translation.

**Output Structure:**
```
ablation_data/output_no_isolength/
└── {category}/
    └── {layout}/
        └── {paper_id}/
            ├── {id}.json
            └── {id}.pdf
```

**Usage:**
```bash
# Run no-isolength experiment
python ablation/run_no_isolength_ablation.py

# Process specific category
python ablation/run_no_isolength_ablation.py --category General
```

## Input Data

Input data is located at `ablation_data/input/` with structure:
```
ablation_data/input/
├── Asset/           # Papers with figures/tables
├── Formula/         # Papers with mathematical formulas
└── General/         # General academic papers
    ├── SingleColunm/
    ├── DoubleColunm/
    └── Complex/
```

Each paper directory contains:
- `{paper_id}.pdf` - Original PDF
- `{paper_id}_content_list_customize.json` - Extracted content blocks

## API Configuration

Both scripts use `config.yaml` for API settings:

```yaml
translation:
  api:
    base_url: "https://your-api.com/v1"
    api_key: "your-key"
    model: "your-model"

reflow:
  api:
    base_url: "https://your-api.com/v1"
    api_key: "your-key"
    model: "your-model"
```

## Expected Results

After running both experiments, you can compare:

1. **Visual Fill Rate**: How well translated text fills the bounding box
2. **Layout Compliance**: Percentage of blocks meeting layout requirements at each round
3. **Translation Quality**: Semantic accuracy (iso-length may trade off for fit)

### Key Metrics

| Metric | Iso-Length | No Iso-Length |
|--------|------------|---------------|
| Average Fill Rate | ~94% | Variable (50-150%) |
| Layout Compliance (round 0) | Higher | Lower |
| Reflow Rounds Needed | Fewer | N/A (no reflow) |
