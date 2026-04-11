#!/usr/bin/env python3
"""
PDF Translation Pipeline
========================
Translates academic papers from English to Chinese with layout preservation.

Usage:
    python run_translation.py                    # Use default config.yaml
    python run_translation.py --config my.yaml   # Use custom config
    python run_translation.py --category formula # Process only formula papers
    python run_translation.py --category 一般论文 --layout 单列
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
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "tools"))


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file with environment variable expansion."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables in API keys
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


def validate_config(config: Dict) -> bool:
    """Validate that required API settings are configured."""
    errors = []
    
    # Check translation API
    trans_api = config.get('translation', {}).get('api', {})
    if not trans_api.get('base_url'):
        errors.append("translation.api.base_url is not configured")
    if not trans_api.get('api_key'):
        errors.append("translation.api.api_key is not configured")
    if not trans_api.get('model'):
        errors.append("translation.api.model is not configured")
    
    # Check reflow API (only if reflow is enabled)
    if config.get('reflow', {}).get('enabled', True):
        reflow_api = config.get('reflow', {}).get('api', {})
        if not reflow_api.get('base_url'):
            errors.append("reflow.api.base_url is not configured")
        if not reflow_api.get('api_key'):
            errors.append("reflow.api.api_key is not configured")
        if not reflow_api.get('model'):
            errors.append("reflow.api.model is not configured")
    
    if errors:
        print("❌ Configuration errors:")
        for err in errors:
            print(f"   - {err}")
        print("\n📝 Please edit config.yaml and fill in the required API settings.")
        return False
    
    return True


def extract_rich_text(data: List[Dict], pdf_path: str) -> List[Dict]:
    """Extract rich text styles from PDF for each text block."""
    doc = fitz.open(pdf_path)
    for item in data:
        if item.get('rich_spans'):
            continue
        bbox = item.get('bbox', [])
        page_num = item.get('page_idx', item.get('page', 0))
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


class TranslationPipeline:
    """Main translation pipeline with configurable options."""

    CATEGORY_ALIASES = {
        'formula': '公式论文',
        'figure': '图表论文',
        'general': '一般论文',
        '公式论文': '公式论文',
        '图表论文': '图表论文',
        '一般论文': '一般论文',
    }

    LAYOUT_ALIASES = {
        'single_column': '单列',
        'double_column': '双列',
        'complex_layout': '复杂布局',
        '单列': '单列',
        '双列': '双列',
        '复杂布局': '复杂布局',
    }
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_dir = BASE_DIR
        self.input_dir = self.base_dir / config['paths']['input_dir']
        self.output_dir = self.base_dir / config['paths']['output_dir']
        from ParseAgent import ParseAgent
        self.parse_agent = ParseAgent(config=config, base_dir=self.base_dir)
        
        # Initialize API clients
        self._init_clients()
        
    def _init_clients(self):
        """Initialize API clients based on configuration."""
        from api_client import APIClient
        
        trans_config = self.config['translation']['api']
        self.translation_client = APIClient(
            api_key=trans_config['api_key'],
            base_url=trans_config['base_url'],
            model=trans_config['model']
        )
        
        reflow_config = self.config['reflow']['api']
        self.reflow_client = APIClient(
            api_key=reflow_config['api_key'],
            base_url=reflow_config['base_url'],
            model=reflow_config['model']
        )

    def _resolve_requested_dirs(self, requested: Optional[List[str]], available: List[str], aliases: Dict[str, str]) -> List[str]:
        if not available:
            return []

        if requested is None:
            return available

        resolved = []
        for name in requested:
            candidate = aliases.get(name, name)
            if candidate in available and candidate not in resolved:
                resolved.append(candidate)
            elif name in available and name not in resolved:
                resolved.append(name)
            else:
                print(f"⚠️  Requested directory not found: {name}")
        return resolved

    def _find_document_assets(self, doc_dir: Path, doc_id: str) -> Optional[Dict]:
        return self.parse_agent.ensure_document_assets(doc_dir, doc_id)
    
    def collect_papers(self, 
                       categories: Optional[List[str]] = None,
                       layouts: Optional[List[str]] = None) -> List[Dict]:
        """Collect all papers to process."""
        papers = []

        if not self.input_dir.exists():
            print(f"❌ Input directory not found: {self.input_dir}")
            return papers

        available_categories = sorted([d.name for d in self.input_dir.iterdir() if d.is_dir()])
        if categories is None:
            categories = self.config['processing'].get('categories')
        resolved_categories = self._resolve_requested_dirs(categories, available_categories, self.CATEGORY_ALIASES)
        if not resolved_categories:
            resolved_categories = available_categories

        for category in resolved_categories:
            category_dir = self.input_dir / category
            if not category_dir.exists():
                continue

            available_layouts = sorted([d.name for d in category_dir.iterdir() if d.is_dir()])
            requested_layouts = layouts if layouts is not None else self.config['processing'].get('layouts')
            resolved_layouts = self._resolve_requested_dirs(requested_layouts, available_layouts, self.LAYOUT_ALIASES)
            if not resolved_layouts:
                resolved_layouts = available_layouts

            for layout in resolved_layouts:
                layout_dir = category_dir / layout
                if not layout_dir.exists():
                    continue

                for doc_dir in layout_dir.iterdir():
                    if not doc_dir.is_dir():
                        continue
                    doc_id = doc_dir.name
                    assets = self._find_document_assets(doc_dir, doc_id)
                    if not assets:
                        continue

                    papers.append({
                        'doc_id': doc_id,
                        'category': category,
                        'layout': layout,
                        'input_dir': assets['input_dir'],
                        'pdf_path': assets['pdf_path'],
                        'json_path': assets['json_path']
                    })
        
        return papers

    def _summarize_translation_guards(self, data: List[Dict]) -> Dict:
        by_lid = {}
        for item in data:
            lid = item.get('logical_para_id')
            guard = item.get('translation_guard') or {}
            if lid is None or not guard:
                continue
            entry = by_lid.setdefault(
                str(lid),
                {
                    'fallback_used': False,
                    'issues': set(),
                },
            )
            entry['fallback_used'] = entry['fallback_used'] or bool(guard.get('fallback_used'))
            for issue in guard.get('issues', []):
                entry['issues'].add(issue)

        total_blocks = max(1, len(by_lid))
        fallback_blocks = sum(1 for value in by_lid.values() if value['fallback_used'])
        return {
            'total_guarded_blocks': len(by_lid),
            'fallback_blocks': fallback_blocks,
            'fallback_ratio': fallback_blocks / total_blocks if total_blocks else 0.0,
        }

    def _should_disable_rewrite_initially(self, planner_agent, data: List[Dict]) -> Tuple[bool, str]:
        if planner_agent and planner_agent.should_disable_rewrite_globally():
            return True, "planner_document_policy"

        guard_summary = self._summarize_translation_guards(data)
        if guard_summary['fallback_blocks'] >= 3 or guard_summary['fallback_ratio'] >= 0.08:
            return True, "translation_formula_guard"

        return False, ""
    
    def process_paper(self, paper: Dict) -> Dict:
        """Process a single paper through the translation pipeline."""
        doc_id = paper['doc_id']
        category = paper['category']
        layout = paper['layout']
        
        result = {
            'doc_id': doc_id,
            'category': category,
            'layout': layout,
            'success': False,
            'message': '',
            'stats': {}
        }
        
        try:
            start_time = datetime.now()
            print(f"\n{'='*70}")
            print(f"🚀 [{doc_id}] Processing - {category}/{layout}")
            print(f"{'='*70}")
            
            # Setup paths
            output_dir = self.output_dir / category / layout / doc_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            pdf_path = paper['pdf_path']
            json_path = paper['json_path']
            
            # Skip if already processed
            output_pdf = output_dir / f"{doc_id}_translated.pdf"
            if self.config['processing'].get('skip_existing', True) and output_pdf.exists():
                print(f"⏭️  [{doc_id}] Already processed, skipping...")
                result['success'] = True
                result['message'] = "Skipped (already exists)"
                return result
            
            # Load content data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract rich text styles
            print(f"📝 [{doc_id}] Extracting rich text styles...")
            data = extract_rich_text(data, str(pdf_path))
            
            # Prepare data for translation
            work_data = copy.deepcopy(data)
            for item in work_data:
                item['translated'] = ''
                if 'golden_style' in item:
                    del item['golden_style']
            
            # Translation
            enable_isolength = self.config['translation'].get('enable_isolength', True)
            m0_plain_translation = self.config['translation'].get('m0_plain_translation', True)
            target_lang = self.config['translation'].get('target_language', 'zh')
            from PlannerAgent import PlannerAgent
            planner_agent = PlannerAgent(
                target_lang=target_lang,
                style_policy=self.config.get('planner', {}).get('style_policy'),
            )
            
            if m0_plain_translation:
                from TranslationAgentNoIsoLength import TranslationAgentNoIsoLength
                print(f"🌍 [{doc_id}] Translating (M0 plain translation)...")
                agent = TranslationAgentNoIsoLength(self.translation_client, target_lang=target_lang, planner=planner_agent)
            elif enable_isolength:
                from TranslationAgent import TranslationAgent
                print(f"🌍 [{doc_id}] Translating (with Iso-Length)...")
                agent = TranslationAgent(self.translation_client, target_lang=target_lang, planner=planner_agent)
            else:
                from TranslationAgentNoIsoLength import TranslationAgentNoIsoLength
                print(f"🌍 [{doc_id}] Translating (NO Iso-Length)...")
                agent = TranslationAgentNoIsoLength(self.translation_client, target_lang=target_lang, planner=planner_agent)
            
            work_data = agent.process_documents(work_data, str(pdf_path))
            
            # Save translated JSON
            if self.config['processing'].get('save_intermediate', True):
                translated_json = output_dir / f"{doc_id}_translated.json"
                with open(translated_json, 'w', encoding='utf-8') as f:
                    json.dump(work_data, f, ensure_ascii=False, indent=2)
            
            # Reflow and rendering
            from ReflowAgent import ReflowAgent
            operation_policy = dict(self.config.get('ablation', {}).get('operation_policy', {}))
            reflow_agent = ReflowAgent(
                self.reflow_client,
                target_lang=target_lang,
                planner=planner_agent,
                operation_policy=operation_policy or None,
                enable_column_joint_optimization=self.config.get('planner', {}).get('enable_column_joint_optimization', True),
            )
            
            reflow_enabled = self.config['reflow'].get('enabled', True)
            max_rounds = self.config['reflow'].get('max_rounds', 3) if reflow_enabled else 1
            enable_rewrite = reflow_enabled
            disable_rewrite, disable_reason = self._should_disable_rewrite_initially(planner_agent, work_data)
            if disable_rewrite:
                enable_rewrite = False
                operation_policy['allow_rewriting'] = False
                reflow_agent.operation_policy['allow_rewriting'] = False
                print(f"🛡️ [{doc_id}] Conservative reflow enabled: rewrite disabled ({disable_reason})")
            
            print(f"📄 [{doc_id}] Rendering (total_round_budget_including_m0={max_rounds})...")
            stats = reflow_agent.run_reflow_task_with_data(
                work_data,
                str(pdf_path),
                str(output_pdf),
                enable_rewrite=enable_rewrite,
                max_rounds=max_rounds
            )

            planner_json = output_dir / f"{doc_id}_planner.json"
            with open(planner_json, 'w', encoding='utf-8') as f:
                json.dump(planner_agent.export_summary(), f, ensure_ascii=False, indent=2)
            parse_manifest = output_dir / f"{doc_id}_parse_manifest.json"
            self.parse_agent.export_parse_manifest(
                paper['input_dir'].parent if paper['input_dir'].name == self.parse_agent.output_subdir else paper['input_dir'],
                doc_id,
                parse_manifest,
            )
            
            # Calculate elapsed time
            elapsed = (datetime.now() - start_time).total_seconds()
            
            result['success'] = True
            result['message'] = f"Success ({elapsed:.0f}s)"
            result['stats'] = stats
            result['planner'] = str(planner_json)
            result['parse_manifest'] = str(parse_manifest)
            print(f"✅ [{doc_id}] Completed in {elapsed:.0f}s")
            
        except Exception as e:
            result['message'] = f"Error: {str(e)}"
            print(f"❌ [{doc_id}] {result['message']}")
            traceback.print_exc()
        
        return result
    
    def run(self, 
            categories: Optional[List[str]] = None,
            layouts: Optional[List[str]] = None,
            max_workers: int = 1) -> List[Dict]:
        """Run the translation pipeline."""
        print("=" * 80)
        print("📚 PDF Translation Pipeline")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Iso-Length: {self.config['translation'].get('enable_isolength', True)}")
        print(f"Max Total Rounds (incl. M0): {self.config['reflow'].get('max_rounds', 3)}")
        print()
        
        # Collect papers
        papers = self.collect_papers(categories, layouts)
        print(f"📊 Papers to process: {len(papers)}")
        
        if not papers:
            print("❌ No papers found!")
            return []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process papers
        results = []
        
        if max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.process_paper, p): p for p in papers}
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            # Sequential processing
            for paper in papers:
                results.append(self.process_paper(paper))
        
        # Save results
        results_file = self.output_dir / "translation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': results,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'enable_isolength': self.config['translation'].get('enable_isolength'),
                    'max_reflow_rounds': self.config['reflow'].get('max_rounds'),
                }
            }, f, ensure_ascii=False, indent=2)
        
        # Print summary
        success_count = sum(1 for r in results if r['success'])
        print("\n" + "=" * 80)
        print("📈 Pipeline Complete")
        print("=" * 80)
        print(f"Success: {success_count}/{len(papers)}")
        print(f"Results: {results_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='PDF Translation Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--category', nargs='+', help='Categories to process (formula, figure, general, or Chinese directory names)')
    parser.add_argument('--layout', nargs='+', help='Layouts to process (single_column, double_column, complex_layout, or Chinese directory names)')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    args = parser.parse_args()
    
    # Load config
    config_path = BASE_DIR / args.config
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Validate config
    if not validate_config(config):
        sys.exit(1)
    
    # Run pipeline
    pipeline = TranslationPipeline(config)
    pipeline.run(
        categories=args.category,
        layouts=args.layout,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
