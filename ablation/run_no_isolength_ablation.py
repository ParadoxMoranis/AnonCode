#!/usr/bin/env python3
"""
Ablation Experiment - No Iso-Length Translation (No Fixed-Length Optimization)
===============================================================================
Runs translation WITHOUT fixed-length optimization.
Forced reflow_max = 0 (no reflow iterations).

This serves as a control group to demonstrate the effectiveness of iso-length translation.

Output structure:
    ablation_data/output_no_isolength/
    └── {category}/
        └── {layout}/
            └── {paper_id}/
                ├── {paper_id}.json
                └── {paper_id}.pdf

Usage:
    python ablation/run_no_isolength_ablation.py
    python ablation/run_no_isolength_ablation.py --category Formula
"""

import os
import sys
import json
import yaml
import copy
import fitz
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "tools"))

INPUT_ROOT = BASE_DIR / "ablation_data" / "input"
OUTPUT_ROOT = BASE_DIR / "ablation_data" / "output_no_isolength"


def load_config() -> Dict:
    """Load API configuration from config.yaml."""
    config_path = BASE_DIR / "config.yaml"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    def expand_env(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, "")
        return value
    
    def expand_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                expand_dict(value)
            elif isinstance(value, str):
                d[key] = expand_env(value)
        return d
    
    return expand_dict(config)


def validate_api_config(config: Dict) -> bool:
    """Validate API configuration."""
    errors = []
    
    trans_api = config.get('translation', {}).get('api', {})
    if not trans_api.get('base_url'):
        errors.append("translation.api.base_url not configured")
    if not trans_api.get('api_key'):
        errors.append("translation.api.api_key not configured")
    if not trans_api.get('model'):
        errors.append("translation.api.model not configured")
    
    # Reflow API is also needed for rendering (even without reflow iterations)
    reflow_api = config.get('reflow', {}).get('api', {})
    if not reflow_api.get('base_url'):
        errors.append("reflow.api.base_url not configured")
    if not reflow_api.get('api_key'):
        errors.append("reflow.api.api_key not configured")
    if not reflow_api.get('model'):
        errors.append("reflow.api.model not configured")
    
    if errors:
        print("❌ Configuration errors:")
        for err in errors:
            print(f"   - {err}")
        print("\n📝 Please edit config.yaml")
        return False
    return True


def extract_rich_text(data: List[Dict], pdf_path: str) -> List[Dict]:
    """Extract rich text styles from PDF."""
    doc = fitz.open(pdf_path)
    for item in data:
        bbox = item.get('bbox', [])
        page_num = item.get('page', 0)
        if len(bbox) != 4 or page_num >= len(doc):
            item['rich_spans'] = []
            continue
        try:
            page = doc[page_num]
            rect = fitz.Rect(bbox)
            blocks = page.get_text('dict', clip=rect)['blocks']
            spans = []
            for block in blocks:
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        spans.append({
                            'text': span['text'],
                            'font': span['font'],
                            'size': span['size'],
                            'flags': span['flags'],
                            'color': span['color']
                        })
            item['rich_spans'] = spans
        except:
            item['rich_spans'] = []
    doc.close()
    return data


def collect_papers(categories: Optional[List[str]] = None) -> List[Dict]:
    """Collect all papers from ablation input directory."""
    papers = []
    
    if categories is None:
        categories = [d.name for d in INPUT_ROOT.iterdir() if d.is_dir()]
    
    for category in categories:
        category_dir = INPUT_ROOT / category
        if not category_dir.exists():
            continue
        
        for layout_dir in category_dir.iterdir():
            if not layout_dir.is_dir():
                continue
            layout = layout_dir.name
            
            for doc_dir in layout_dir.iterdir():
                if not doc_dir.is_dir():
                    continue
                doc_id = doc_dir.name
                pdf_path = doc_dir / f"{doc_id}.pdf"
                json_path = doc_dir / f"{doc_id}_content_list_customize.json"
                
                if pdf_path.exists() and json_path.exists():
                    papers.append({
                        'doc_id': doc_id,
                        'category': category,
                        'layout': layout,
                        'input_dir': doc_dir,
                        'pdf_path': pdf_path,
                        'json_path': json_path
                    })
    
    return papers


def process_paper(paper: Dict, config: Dict) -> Dict:
    """Process a single paper WITHOUT iso-length translation and NO reflow."""
    from api_client import APIClient
    from PlannerAgent import PlannerAgent
    from TranslationAgentNoIsoLength import TranslationAgentNoIsoLength
    from ReflowAgent import ReflowAgent
    
    doc_id = paper['doc_id']
    category = paper['category']
    layout = paper['layout']
    
    result = {
        'doc_id': doc_id,
        'category': category,
        'layout': layout,
        'success': False,
        'message': ''
    }
    
    try:
        start_time = datetime.now()
        print(f"\n{'='*70}")
        print(f"🚀 [{doc_id}] No Iso-Length Ablation - {category}/{layout}")
        print(f"   Mode: NO Iso-Length, NO Reflow")
        print(f"{'='*70}")
        
        # Setup output directory
        output_dir = OUTPUT_ROOT / category / layout / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        with open(paper['json_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract rich text
        print(f"📝 [{doc_id}] Extracting rich text styles...")
        data = extract_rich_text(data, str(paper['pdf_path']))
        
        # Prepare data
        work_data = copy.deepcopy(data)
        for item in work_data:
            item['translated'] = ''
            if 'golden_style' in item:
                del item['golden_style']
        
        # Initialize API clients
        trans_api = config['translation']['api']
        translation_client = APIClient(
            api_key=trans_api['api_key'],
            base_url=trans_api['base_url'],
            model=trans_api['model']
        )
        
        reflow_api = config['reflow']['api']
        reflow_client = APIClient(
            api_key=reflow_api['api_key'],
            base_url=reflow_api['base_url'],
            model=reflow_api['model']
        )
        
        # Translation WITHOUT iso-length
        print(f"🌍 [{doc_id}] Translating (NO Iso-Length)...")
        planner_agent = PlannerAgent(target_lang='zh')
        translation_agent = TranslationAgentNoIsoLength(translation_client, target_lang='zh', planner=planner_agent)
        work_data = translation_agent.process_documents(work_data, str(paper['pdf_path']))
        
        # Save translated JSON
        output_json = output_dir / f"{doc_id}.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(work_data, f, ensure_ascii=False, indent=2)
        
        # Render PDF (NO reflow - max_rounds=0)
        print(f"📄 [{doc_id}] Rendering (NO Reflow)...")
        output_pdf = output_dir / f"{doc_id}.pdf"
        
        reflow_agent = ReflowAgent(reflow_client, target_lang='zh', planner=planner_agent)
        stats = reflow_agent.run_reflow_task_with_data(
            work_data,
            str(paper['pdf_path']),
            str(output_pdf),
            enable_rewrite=False,  # No rewrite/reflow
            max_rounds=0           # Forced: no reflow rounds
        )

        planner_json = output_dir / f"{doc_id}_planner.json"
        with open(planner_json, 'w', encoding='utf-8') as f:
            json.dump(planner_agent.export_summary(), f, ensure_ascii=False, indent=2)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        result['success'] = True
        result['message'] = f"Success ({elapsed:.0f}s)"
        result['output_json'] = str(output_json)
        result['output_pdf'] = str(output_pdf)
        result['planner'] = str(planner_json)
        result['stats'] = stats
        print(f"✅ [{doc_id}] Completed in {elapsed:.0f}s")
        
    except Exception as e:
        result['message'] = f"Error: {str(e)}"
        print(f"❌ [{doc_id}] {result['message']}")
        traceback.print_exc()
    
    return result


def main():
    parser = argparse.ArgumentParser(description='No Iso-Length Ablation Experiment')
    parser.add_argument('--category', nargs='+', help='Categories to process (Asset, Formula, General)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔬 Ablation Experiment: NO Iso-Length Translation")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: NO Iso-Length, NO Reflow (forced max_rounds=0)")
    print(f"Output: {OUTPUT_ROOT}")
    print()
    
    # Load and validate config
    config = load_config()
    if not validate_api_config(config):
        sys.exit(1)
    
    # Collect papers
    papers = collect_papers(args.category)
    print(f"📊 Papers to process: {len(papers)}")
    
    if not papers:
        print("❌ No papers found!")
        return
    
    # Process papers
    results = []
    for paper in papers:
        result = process_paper(paper, config)
        results.append(result)
    
    # Save experiment results
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_ROOT / "no_isolength_experiment_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'no_isolength_ablation',
            'mode': 'NO Iso-Length, NO Reflow',
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)
    
    # Summary
    success_count = sum(1 for r in results if r['success'])
    print("\n" + "=" * 80)
    print("📈 Experiment Complete")
    print("=" * 80)
    print(f"Success: {success_count}/{len(papers)}")
    print(f"Results: {results_file}")


if __name__ == "__main__":
    main()
