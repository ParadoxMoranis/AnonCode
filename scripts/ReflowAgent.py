import os
import json
import re
import fitz
from tqdm import tqdm
from typing import Dict, List, Tuple
from tools.pdf_reflow_tool import PDFReflowTool
from tools.api_client import APIClient
from tools.text_refiner import TextRefiner

class ReflowAgent:
    def __init__(self, api_client: APIClient, target_lang='zh', planner=None, operation_policy=None, enable_column_joint_optimization=False):
        self.target_lang = target_lang
        self.client = api_client
        self.planner = planner
        self.refiner = TextRefiner(self.client)
        self.tool = PDFReflowTool(lang=target_lang)
        self.operation_policy = operation_policy or {
            'allow_rewriting': True,
            'allow_line_spacing': True,
            'allow_char_spacing': True,
            'allow_font_size': True,
        }
        self.enable_column_joint_optimization = enable_column_joint_optimization
        
        # [关键修复]
        # 1. 移除了 'header' (原指页眉)，防止误杀 'section_header' (一级标题)
        # 2. 改为 'page_header'，更加精确
        # 3. 明确包含 'image' 和 'table'，确保不重绘它们
        self.PRESERVE_TYPES = {
            'author', 'affiliation', 'reference', 'equation', 
            'figure', 'table', 'code', 'url', 'page_header', 'page_footer',
            'image', 'rect', 'curve', 'line',
            'aside_text', 'page_number',
        }
        self.global_golden_config = {}
        
        # 样式对齐缓存，避免重复调用 LLM
        self._style_align_cache = {}
        if self._is_font_size_locked():
            self.operation_policy['allow_font_size'] = False

    def _matches_preserve_type(self, raw_type: str, detected_type: str = '') -> bool:
        raw = (raw_type or '').lower()
        detected = (detected_type or '').lower()
        for preserve_type in self.PRESERVE_TYPES:
            if preserve_type in raw or preserve_type in detected:
                return True
        return False

    def _is_font_size_locked(self) -> bool:
        if self.planner:
            return bool(self.planner.style_policy.get('lock_font_size', False))
        return False

    def _layout_mode(self) -> str:
        if self.planner:
            return self.planner.document_profile.get('layout_mode', 'single_col')
        return 'single_col'

    def _document_metric(self, key: str, default: float = 0.0) -> float:
        if not self.planner:
            return default
        try:
            return float(self.planner.document_profile.get(key, default))
        except Exception:
            return default

    def _is_formula_dense_document(self) -> bool:
        if not self.planner:
            return False
        profile = self.planner.document_profile
        if profile.get('conservative_rewrite_mode') or profile.get('disable_global_rewrite'):
            return True
        math_ratio = self._document_metric('math_like_block_ratio', 0.0)
        multi_box_ratio = self._document_metric('multi_box_block_ratio', 0.0)
        theorem_ratio = self._document_metric('theorem_like_block_ratio', 0.0)
        return math_ratio >= 0.16 or multi_box_ratio >= 0.18 or theorem_ratio >= 0.14

    def _is_prose_heavy_document(self) -> bool:
        if not self.planner:
            return False
        if self._is_formula_dense_document():
            return False
        math_ratio = self._document_metric('math_like_block_ratio', 0.0)
        multi_box_ratio = self._document_metric('multi_box_block_ratio', 0.0)
        median_density = self._document_metric('median_source_density', 0.0)
        avg_text_length = self._document_metric('avg_text_length', 0.0)
        return (
            math_ratio <= 0.06
            and multi_box_ratio <= 0.08
            and median_density >= 58.0
            and avg_text_length >= 380.0
        )

    def _underflow_threshold(self, font_key: str = '', is_title: bool = False) -> float:
        if is_title:
            return 0.66
        if self._layout_mode() == 'double_col' and font_key == 'body':
            if self._is_formula_dense_document():
                return 0.78
            if self._is_prose_heavy_document():
                return 0.88
            return 0.84
        if self._layout_mode() == 'double_col':
            if self._is_formula_dense_document():
                return 0.74
            if self._is_prose_heavy_document():
                return 0.82
            return 0.78
        if self._layout_mode() == 'single_col' and font_key == 'body':
            if self._is_formula_dense_document():
                return 0.89
            return 0.92 if self._is_prose_heavy_document() else 0.90
        if self._layout_mode() == 'single_col':
            return 0.82
        return 0.68

    def _rewrite_fill_target(self, issue: str, font_key: str = '') -> float:
        layout_mode = self._layout_mode()
        if issue == 'OVERFLOW':
            return 0.985 if layout_mode == 'single_col' else 0.97
        if layout_mode == 'double_col' and font_key == 'body':
            return 0.95
        if layout_mode == 'double_col':
            return 0.94
        if layout_mode == 'single_col' and font_key == 'body':
            return 0.985 if not self._is_formula_dense_document() else 0.97
        if layout_mode == 'single_col':
            return 0.95
        return 0.92

    def _max_underflow_rewrite_attempts(self, item: Dict, font_key: str = '') -> int:
        if font_key != 'body':
            return 1
        if self._layout_mode() == 'single_col':
            guard = item.get('translation_guard') or {}
            if guard.get('fallback_used'):
                return 1
            risk_flags = set((self.planner.block_states.get(str(item.get('logical_para_id')), {}).get('meta', {}) or {}).get('risk_flags', [])) if self.planner else set()
            if {'multi_box', 'contains_math', 'contains_latex_commands', 'theorem_like'} & risk_flags:
                return 1
            return 2
        return 1

    def _math_segment_count(self, text: str) -> int:
        if not text:
            return 0
        inline_pairs = text.count('$') // 2
        display_pairs = text.count('\\[') + text.count('\\(')
        return inline_pairs + display_pairs

    def _safe_fill_limit(self, bbox: List[float], text: str, font_key: str = '') -> float:
        is_body = font_key == 'body'
        layout_mode = self._layout_mode()
        if font_key in ('title', 'heading'):
            limit = 0.985
        elif layout_mode == 'single_col' and is_body:
            limit = 0.965 if self._is_prose_heavy_document() else 0.955
        elif is_body:
            limit = 0.965 if self._is_prose_heavy_document() else 0.955
        else:
            limit = 0.97

        compact_chars = len((text or '').replace(' ', '').replace('\n', ''))
        math_segments = self._math_segment_count(text or '')
        if compact_chars >= 80 or math_segments >= 2:
            limit -= 0.025
        if len(bbox) == 4:
            box_height = float(bbox[3]) - float(bbox[1])
            if box_height <= 18:
                limit -= 0.02

        return max(0.84, min(0.985, limit))

    def _candidate_fill_score(self, metrics: Dict, target_fill: float, safe_fill_limit: float) -> float:
        fill_ratio = float(metrics.get('fill_ratio', 0.0))
        if metrics.get('is_overflow') or fill_ratio > safe_fill_limit + 1e-6:
            return -1e9
        bonus = min(fill_ratio, safe_fill_limit)
        closeness = max(0.0, 1.0 - abs(target_fill - fill_ratio) * 8.0)
        return bonus * 1000.0 + closeness * 25.0

    def _rewrite_allowed_for_block(self, item: Dict, lid, text: str, font_key: str, detected_type: str = '') -> bool:
        if not self.operation_policy.get('allow_rewriting', True):
            return False
        if not text or lid is None:
            return False
        if self.planner and not self.planner.can_rewrite_block(lid):
            return False
        guard = item.get('translation_guard') or {}
        if guard.get('fallback_used'):
            return False
        raw_type = (item.get('type') or '').lower()
        detected = (detected_type or raw_type).lower()
        if self._matches_preserve_type(raw_type, detected):
            return False
        if font_key in ('title', 'heading') or 'title' in detected or 'section_header' in detected:
            return False
        return True

    def _apply_force_fit_without_font_change(self, final_style: Dict, golden_style: Dict, overflow_px: float) -> Tuple[Dict, bool, List[str], str]:
        if overflow_px <= 5:
            return final_style, True, [], 'accept_minor_overflow'

        adjusted_style = final_style.copy()
        adjusted_style['line'] = max(1.15, adjusted_style['line'] - 0.08)
        adjusted_style['char'] = max(
            self._get_char_spacing_guardrails(golden_style.get('char', 0.05))['global_min'],
            adjusted_style.get('char', golden_style.get('char', 0.05)) - 0.01,
        )
        return adjusted_style, True, ['LineSpacing', 'CharSpacing'], 'force_fit_without_font_change'

    def _get_char_spacing_guardrails(self, base_char: float) -> Dict[str, float]:
        default_center = round(base_char, 3)
        default = {
            'global_min': max(0.0, default_center - 0.02),
            'global_max': min(0.12, default_center + 0.02),
            'center': default_center,
            'max_delta_from_bucket': 0.02,
        }
        if self.planner:
            guardrails = self.planner.global_plan.get('char_spacing_guardrails')
            if isinstance(guardrails, dict):
                merged = default.copy()
                merged.update(guardrails)
                return merged
        return default

    def _clamp_style_guardrails(self, style: Dict, base_style: Dict) -> Dict:
        clamped = style.copy()
        guardrails = self._get_char_spacing_guardrails(base_style.get('char', 0.05))
        base_char = float(base_style.get('char', guardrails.get('center', 0.05)))
        max_delta = guardrails.get('max_delta_from_bucket', 0.02)
        local_min = max(guardrails['global_min'], base_char - max_delta)
        local_max = min(guardrails['global_max'], base_char + max_delta)
        clamped['char'] = min(local_max, max(local_min, clamped.get('char', base_char)))
        if self._is_font_size_locked():
            clamped['size'] = base_style.get('size', clamped.get('size', 10.5))
            clamped['_lock_font_size'] = True
        return clamped

    def _build_column_groups(self, data: List[Dict]) -> Dict[str, List[int]]:
        groups = {}
        for idx, item in enumerate(data):
            raw_type = item.get('type', '').lower()
            if raw_type in ['image', 'table', 'figure']:
                continue
            bbox = item.get('bbox', [])
            if len(bbox) != 4:
                continue
            page_idx = item.get('page_idx', item.get('page', 0))
            center_x = (bbox[0] + bbox[2]) / 2
            page_width = self.planner.document_profile.get('page_width', 0.0) if self.planner else 0.0
            if page_width > 0 and center_x < page_width * 0.42:
                col = 'left'
            elif page_width > 0 and center_x > page_width * 0.58:
                col = 'right'
            else:
                col = 'full'
            key = f"{page_idx}:{col}"
            groups.setdefault(key, []).append(idx)
        return groups

    def _prepare_column_joint_plan(self, data: List[Dict]) -> Dict[int, Dict]:
        if not self.enable_column_joint_optimization:
            return {}
        if self._layout_mode() == 'single_col':
            return {}

        column_groups = self._build_column_groups(data)
        plan = {}
        for column_id, indices in column_groups.items():
            entries = []
            for idx in indices:
                item = data[idx]
                text = item.get('translated', '')
                if not text:
                    continue
                golden_style = item.get('golden_style', {'size': 9.0, 'line': 1.35, 'char': 0.05})
                style_keys = ['size', 'line', 'char', 'font_key', 'is_bold', 'is_italic', 'text_color', 'bg_color']
                base_style = {k: v for k, v in golden_style.items() if k in style_keys}
                if 'char' not in base_style:
                    base_style['char'] = 0.05
                metrics = self.tool.simulate_layout_metrics(item['bbox'], text, base_style)
                fill_ratio = metrics.get('fill_ratio', 1.0)
                issue = None
                if metrics.get('is_overflow') or fill_ratio > 1.0:
                    issue = 'OVERFLOW'
                elif fill_ratio < self._underflow_threshold(base_style.get('font_key', 'body'), False):
                    issue = 'UNDERFLOW'
                if issue:
                    entries.append(
                        {
                            'index': idx,
                            'lid': item.get('logical_para_id'),
                            'issue': issue,
                            'fill_ratio': fill_ratio,
                            'metrics': metrics,
                            'current_chars': len(text.replace(' ', '').replace('\n', '')),
                        }
                    )
            overflow_entries = [entry for entry in entries if entry['issue'] == 'OVERFLOW' and entry['lid'] is not None]
            underflow_entries = [entry for entry in entries if entry['issue'] == 'UNDERFLOW' and entry['lid'] is not None]
            pair_count = min(len(overflow_entries), len(underflow_entries))
            for pair_idx in range(pair_count):
                overflow_entry = overflow_entries[pair_idx]
                underflow_entry = underflow_entries[pair_idx]
                overflow_target = max(1, int(overflow_entry['current_chars'] * 0.9))
                underflow_target = max(underflow_entry['current_chars'] + 6, int(underflow_entry['current_chars'] * 1.12))
                plan[overflow_entry['lid']] = {
                    'column_id': column_id,
                    'partner_lid': underflow_entry['lid'],
                    'mode': 'shorten',
                    'target_chars': overflow_target,
                }
                plan[underflow_entry['lid']] = {
                    'column_id': column_id,
                    'partner_lid': overflow_entry['lid'],
                    'mode': 'lengthen',
                    'target_chars': underflow_target,
                }
        return plan

    def _horizontal_overlap_width(self, bbox_a: List[float], bbox_b: List[float]) -> float:
        if len(bbox_a) != 4 or len(bbox_b) != 4:
            return 0.0
        return max(0.0, min(float(bbox_a[2]), float(bbox_b[2])) - max(float(bbox_a[0]), float(bbox_b[0])))

    def _compute_render_bottom_guards(self, data: List[Dict]) -> Dict[int, float]:
        page_blocks: Dict[int, List[Tuple[int, List[float]]]] = {}
        for idx, item in enumerate(data):
            bbox = item.get('bbox', [])
            if len(bbox) != 4:
                continue
            page_idx = item.get('page_idx', item.get('page', 0))
            page_blocks.setdefault(page_idx, []).append((idx, bbox))

        guards: Dict[int, float] = {}
        spill_budget = 3.0
        gap_pad = 0.8

        for page_idx, blocks in page_blocks.items():
            blocks.sort(key=lambda entry: (entry[1][1], entry[1][0]))
            for pos, (idx, bbox) in enumerate(blocks):
                current_top = float(bbox[1])
                current_bottom = float(bbox[3])
                current_width = max(1.0, float(bbox[2]) - float(bbox[0]))
                default_limit = current_bottom + spill_budget
                limit = default_limit

                for next_idx, next_bbox in blocks[pos + 1:]:
                    next_top = float(next_bbox[1])
                    if next_top > default_limit + 24:
                        break

                    overlap_w = self._horizontal_overlap_width(bbox, next_bbox)
                    min_required_overlap = max(8.0, min(current_width, max(1.0, float(next_bbox[2]) - float(next_bbox[0]))) * 0.18)
                    if overlap_w < min_required_overlap:
                        continue

                    if next_top <= current_top:
                        continue

                    limit = min(limit, max(current_bottom, next_top - gap_pad))
                    break

                guards[idx] = limit

        return guards
    
    def _extract_style_keywords(self, rich_spans: List[Dict]) -> List[Tuple[str, bool, bool]]:
        """从 rich_spans 提取有特殊样式的词"""
        keywords = []
        for span in rich_spans:
            text = span.get('text', '').strip()
            if not text:
                continue
            flags = span.get('flags', 0)
            is_bold = bool(flags & 16)
            is_italic = bool(flags & 2)
            if is_bold or is_italic:
                keywords.append((text, is_bold, is_italic))
        return keywords

    def _normalize_style_token(self, token: str) -> str:
        return " ".join((token or "").strip().split())

    def _is_meaningful_style_token(self, token: str) -> bool:
        normalized = self._normalize_style_token(token)
        if not normalized:
            return False
        if re.fullmatch(r'[\W_]+', normalized, flags=re.UNICODE):
            return False
        visible_chars = [ch for ch in normalized if not ch.isspace()]
        if len(visible_chars) <= 1:
            return False
        if len(visible_chars) <= 2 and re.fullmatch(r'[\u4e00-\u9fff]+', normalized):
            return False
        return True

    def _sanitize_style_mapping(self, translated: str, style_mapping: Dict[str, Tuple[bool, bool]]) -> Dict[str, Tuple[bool, bool]]:
        if not translated or not style_mapping:
            return {}

        cleaned: Dict[str, Tuple[bool, bool]] = {}
        styled_ranges: List[Tuple[int, int]] = []

        for raw_token, flags in style_mapping.items():
            token = self._normalize_style_token(raw_token)
            if not self._is_meaningful_style_token(token):
                continue
            if token not in translated:
                continue

            is_bold = bool(flags[0]) if isinstance(flags, (tuple, list)) and len(flags) > 0 else False
            is_italic = bool(flags[1]) if isinstance(flags, (tuple, list)) and len(flags) > 1 else False
            if not (is_bold or is_italic):
                continue

            occurrences = []
            start = 0
            while True:
                pos = translated.find(token, start)
                if pos == -1:
                    break
                occurrences.append((pos, pos + len(token)))
                start = pos + len(token)

            if not occurrences:
                continue
            if len(token) <= 2 and len(occurrences) > 2:
                continue
            if len(token) <= 3 and len(occurrences) > 3:
                continue

            cleaned[token] = (
                cleaned.get(token, (False, False))[0] or is_bold,
                cleaned.get(token, (False, False))[1] or is_italic,
            )
            styled_ranges.extend(occurrences)

        if not cleaned:
            return {}

        styled_ranges.sort()
        merged_ranges: List[Tuple[int, int]] = []
        for start, end in styled_ranges:
            if not merged_ranges or start > merged_ranges[-1][1]:
                merged_ranges.append((start, end))
            else:
                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))

        styled_chars = sum(max(0, end - start) for start, end in merged_ranges)
        coverage = styled_chars / max(1, len(translated))
        if coverage > 0.72 and len(cleaned) >= 4:
            return {}
        if len(cleaned) > max(14, len(translated) // 6):
            return {}

        return cleaned
    
    def _semantic_align_styles(self, source_text: str, translated: str, style_keywords: List[Tuple[str, bool, bool]]) -> Dict[str, Tuple[bool, bool]]:
        """
        使用 LLM 进行语义对齐，找出样式词在译文中的对应翻译
        
        返回: {译文中的词: (is_bold, is_italic)}
        """
        if not style_keywords:
            return {}
        
        # 生成缓存 key
        cache_key = f"{translated[:50]}_{len(style_keywords)}"
        if cache_key in self._style_align_cache:
            return self._style_align_cache[cache_key]
        
        # 构建 prompt
        style_list = []
        for text, is_bold, is_italic in style_keywords:
            style = []
            if is_bold: style.append("粗体")
            if is_italic: style.append("斜体")
            style_str = "+".join(style)
            style_list.append(f'"{text}" ({style_str})')
        
        prompt = f"""请帮我找出以下样式词在译文中对应的翻译。

原文中有特殊样式的词：
{chr(10).join(style_list)}

译文：
{translated}

请直接返回JSON格式，列出每个样式词在译文中的对应翻译。如果某个词在译文中保持原样（如Figure、Table），则返回原词。
格式: {{"原词1": "译文对应词1", "原词2": "译文对应词2", ...}}

只返回JSON，不要其他内容。如果某个词在译文中找不到对应，可以省略。"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat_completion(messages)
            
            if not response:
                return {}
            
            # 清理响应
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()
            
            # 解析 JSON
            import json as json_module
            mapping = json_module.loads(response)
            
            # 构建结果
            result = {}
            for orig_text, is_bold, is_italic in style_keywords:
                if orig_text in mapping:
                    trans_text = mapping[orig_text]
                    if trans_text and trans_text in translated:
                        result[trans_text] = (is_bold, is_italic)
                # 同时检查原词是否在译文中（如 Figure, Table）
                if orig_text in translated:
                    result[orig_text] = (is_bold, is_italic)

            result = self._sanitize_style_mapping(translated, result)
            self._style_align_cache[cache_key] = result
            return result
            
        except Exception:
            return {}

    def _get_micro_tuned_style(self, bbox: List[float], text: str, base_style: Dict) -> Tuple[Dict, str, float]:
        """
        [V3] 适度调整：行距/字距优先；若黄金字号锁定，则完全禁止缩字号。
        
        核心原则：
        1. 字号只能缩小，最多 -1pt
        2. 行距/字距 ±5% 调整
        3. 填充目标 95%-110%
        """
        base_line = base_style.get('line', 1.35)
        base_char = base_style.get('char', 0.05)
        font_size = base_style.get('size', 10.5)
        font_key = base_style.get('font_key', '')
        char_guardrails = self._get_char_spacing_guardrails(base_char)
        safe_fill_limit = self._safe_fill_limit(bbox, text, font_key)
        underfill_target = self._underflow_threshold(font_key, font_key in ('title', 'heading'))

        # [V4] 调整范围：收紧可略更积极，拉伸可明显更积极，但仍受安全上限约束
        min_line = base_line * 0.93
        max_line = base_line * 1.08
        if self._layout_mode() == 'double_col' and font_key == 'body':
            expand_factor = 1.18 if self._is_formula_dense_document() else (1.28 if self._is_prose_heavy_document() else 1.22)
        elif self._layout_mode() == 'single_col' and font_key == 'body':
            expand_factor = 1.14 if self._is_prose_heavy_document() else 1.1
        else:
            expand_factor = 1.12
        expand_line = base_line * expand_factor
        
        char_flex = font_size * 0.035
        local_delta = min(char_flex, char_guardrails.get('max_delta_from_bucket', 0.02))
        min_char = max(char_guardrails['global_min'], base_char - local_delta)
        max_char = min(char_guardrails['global_max'], base_char + local_delta)
        
        # [V3] 字号只能缩小，最多 -1pt；锁字号时完全不允许调整
        font_size_locked = self._is_font_size_locked()
        min_font_size = font_size if font_size_locked else max(6, font_size - 1.0)
        
        # 计算基础指标
        metrics = self.tool.simulate_layout_metrics(bbox, text, base_style)
        fill_ratio = metrics['fill_ratio']

        # [V3] 使用安全余量，避免模拟“刚好放下”但实际渲染遮盖
        is_acceptable = underfill_target <= fill_ratio <= safe_fill_limit and not metrics['is_overflow']

        # ===== 策略1: 正常范围 -> 保持原样 =====
        if is_acceptable:
            return self._clamp_style_guardrails(base_style, base_style), 'perfect', 0.0

        # ===== 策略1.5: 欠填充 -> 先尝试物理拉伸，减少双列大空白 =====
        if (
            not metrics.get('is_overflow', False)
            and fill_ratio < underfill_target
            and (self.operation_policy.get('allow_line_spacing', True) or self.operation_policy.get('allow_char_spacing', True))
        ):
            best_style = None
            best_metrics = metrics
            best_score = self._candidate_fill_score(metrics, underfill_target, safe_fill_limit)
            line_candidates = [base_line]
            char_candidates = [base_char]
            if self.operation_policy.get('allow_line_spacing', True):
                line_steps = 10 if font_key == 'body' else 7
                for step in range(1, line_steps + 1):
                    line_candidates.append(min(expand_line, base_line * (1 + step * 0.02)))
            if self.operation_policy.get('allow_char_spacing', True):
                char_step = 0.004 if font_key == 'body' else 0.003
                if self._layout_mode() == 'double_col' and font_key == 'body' and self._is_prose_heavy_document():
                    char_step = 0.005
                if self._layout_mode() == 'single_col' and font_key == 'body':
                    char_step = 0.005
                char_steps = 10 if (self._layout_mode() == 'single_col' and font_key == 'body') else (8 if font_key == 'body' else 6)
                for step in range(1, char_steps + 1):
                    char_candidates.append(min(max_char, base_char + step * char_step))

            for cand_line in sorted(set(line_candidates)):
                for cand_char in sorted(set(char_candidates)):
                    if cand_line == base_line and cand_char == base_char:
                        continue
                    expand_style = base_style.copy()
                    expand_style['line'] = cand_line
                    expand_style['char'] = cand_char
                    expand_metrics = self.tool.simulate_layout_metrics(bbox, text, expand_style)
                    candidate_score = self._candidate_fill_score(expand_metrics, underfill_target, safe_fill_limit)
                    if candidate_score > best_score + 1e-6:
                        best_style = expand_style
                        best_metrics = expand_metrics
                        best_score = candidate_score

            if best_style is not None and best_metrics.get('fill_ratio', 0.0) > fill_ratio + 0.004:
                return self._clamp_style_guardrails(best_style, base_style), 'tuned_expand', 0.0

        # ===== 策略2: 溢出 -> 先调整行距/字距 =====
        # 尝试更细粒度的收紧，并优先选择最接近安全上限的候选
        if self.operation_policy.get('allow_line_spacing', True) or self.operation_policy.get('allow_char_spacing', True):
            best_tight = None
            best_tight_score = -1e9
            line_candidates = [base_line]
            char_candidates = [base_char]
            if self.operation_policy.get('allow_line_spacing', True):
                for step in range(0, 8):
                    line_candidates.append(max(min_line, base_line * (1 - step * 0.012)))
            if self.operation_policy.get('allow_char_spacing', True):
                for step in range(0, 8):
                    char_candidates.append(max(min_char, base_char - step * 0.005))

            for cand_line in sorted(set(line_candidates), reverse=True):
                for cand_char in sorted(set(char_candidates), reverse=True):
                    tight_style = base_style.copy()
                    tight_style['line'] = cand_line
                    tight_style['char'] = cand_char
                    m_tight = self.tool.simulate_layout_metrics(bbox, text, tight_style)
                    if m_tight['is_overflow'] or m_tight['fill_ratio'] > safe_fill_limit:
                        continue
                    candidate_score = self._candidate_fill_score(m_tight, min(safe_fill_limit, 0.97), safe_fill_limit)
                    if candidate_score > best_tight_score:
                        best_tight = tight_style
                        best_tight_score = candidate_score
            if best_tight is not None:
                return self._clamp_style_guardrails(best_tight, base_style), 'tuned_compact', 0.0
        
        # ===== 策略3: 行距/字距不够 -> 尝试缩小字号（最多 -1pt）=====
        if self.operation_policy.get('allow_font_size', True) and not font_size_locked:
            for size_step in [0.3, 0.6, 1.0]:
                tight_style = base_style.copy()
                tight_style['size'] = max(min_font_size, font_size - size_step)
                if self.operation_policy.get('allow_line_spacing', True):
                    tight_style['line'] = min_line
                if self.operation_policy.get('allow_char_spacing', True):
                    tight_style['char'] = min_char
                
                m_tight = self.tool.simulate_layout_metrics(bbox, text, tight_style)
                
                if not m_tight['is_overflow'] and m_tight['fill_ratio'] <= safe_fill_limit:
                    return self._clamp_style_guardrails(tight_style, base_style), 'tuned_size_down', 0.0
        
        # ===== 策略4: 仍然溢出 -> 标记 =====
        tight_max = base_style.copy()
        tight_max['size'] = font_size if font_size_locked else min_font_size
        tight_max['line'] = min_line
        tight_max['char'] = min_char
        m_max = self.tool.simulate_layout_metrics(bbox, text, tight_max)
        safe_box_height = m_max['box_height'] * safe_fill_limit
        overflow_amount = max(0.0, m_max['needed_height'] - safe_box_height)
        return self._clamp_style_guardrails(tight_max, base_style), 'overflow', overflow_amount

    def run_reflow_task_with_data(self, data, input_pdf, output_pdf, enable_rewrite=True, max_rounds=4):
        """
        [V10 终极版] 多轮 reflow，最后一轮强制满足。
        `max_rounds` 表示总轮次预算，包含 M0 预排版/模拟阶段。
        
        Args:
            data: 翻译后的数据
            input_pdf: 原始 PDF 路径
            output_pdf: 输出 PDF 路径
            enable_rewrite: 是否启用重写功能（消融实验用）
            max_rounds: 最大总轮数，包含 M0（默认4轮）
        
        Returns:
            dict: 统计信息 {'round_1': count, ..., 'total_blocks': count}
        """
        total_round_budget = max(1, int(max_rounds))
        iterative_rounds = max(1, total_round_budget - 1) if enable_rewrite else 1
        print(
            f"[Reflow] Rendering... "
            f"(rewrite={'ON' if enable_rewrite else 'OFF'}, "
            f"total_round_budget={total_round_budget}, iterative_rounds={iterative_rounds}, m0_inclusive=True)"
        )
        processed_lids = set()

        if self.planner:
            self.planner.ensure_block_registry(data)
        
        # [V10 新增] 统计每轮满足要求的block数量
        round_stats = {
            'total_round_budget_including_m0': total_round_budget,
            'iterative_reflow_rounds': iterative_rounds,
        }
        round_stats.update({f'round_{i+1}': 0 for i in range(iterative_rounds)})
        stable_lids = set()
        underflow_rewrite_attempts = {}

        for round_i in range(iterative_rounds):
            if self.planner:
                self.planner.start_round(round_i + 1)
            column_joint_plan = self._prepare_column_joint_plan(data)
            is_final_round = (round_i == iterative_rounds - 1)  # [V10] 标记最后一轮
            round_label = "Final (Force Satisfy)" if is_final_round else f"{round_i+1}/{iterative_rounds}"
            print(f"\n>> Round {round_label}")
            needs_rewrite = False
            satisfied_this_round = 0
            skipped_count = 0
            rewrite_count = 0
            forced_count = 0  # [V10] 强制满足的block数量
            
            for i in tqdm(range(len(data)), desc="Reflow Check"):
                item = data[i]
                # [关键] 严格检查：如果是 Image/Table，直接跳过 (不清理，不重绘)
                raw_type = item.get('type', '').lower()
                if raw_type in ['image', 'table', 'figure']:
                    continue

                # 检查是否在保留列表中 (Skip text types)
                itype = item.get('detected_type', raw_type).lower()
                if self._matches_preserve_type(raw_type, itype):
                    continue
                
                text = item.get('translated', '')
                if not text: continue
                lid = item.get('logical_para_id')
                
                if self.planner and self.planner.should_skip_block(lid):
                    skipped_count += 1
                    continue
                if not self.planner and lid in stable_lids:
                    skipped_count += 1
                    continue
                
                golden_style = item.get('golden_style')
                if not golden_style: 
                    golden_style = {'size': 9.0, 'line': 1.35, 'char': 0.05}
                
                # [增强] 保留所有样式信息，包括粗体/斜体/颜色/背景色
                style_keys = ['size', 'line', 'char', 'font_key', 'is_bold', 'is_italic', 'text_color', 'bg_color']
                base_style = {k: v for k, v in golden_style.items() if k in style_keys}
                if 'char' not in base_style: base_style['char'] = 0.05
                
                bbox = item['bbox']
                base_metrics = self.tool.simulate_layout_metrics(bbox, text, base_style)
                final_style, status, overflow_px = self._get_micro_tuned_style(bbox, text, base_style)
                final_metrics = self.tool.simulate_layout_metrics(bbox, text, final_style)
                
                action = None
                planner_feedback = None
                issue = 'OPTIMAL'
                decision = 'observe'
                physical_actions = []
                forced_fit = False
                state_before = self.planner.get_block_state(lid) if self.planner else ('OPTIMAL' if lid in stable_lids else 'PENDING')
                joint_hint = column_joint_plan.get(lid)

                if status in {'tuned_compact', 'tuned_expand'}:
                    physical_actions.extend(['LineSpacing', 'CharSpacing'])
                elif status == 'tuned_size_down':
                    physical_actions.extend(['FontSize', 'LineSpacing', 'CharSpacing'])
                
                # [V6 修复] 标题类型永不触发重写，只调整字号
                font_key = golden_style.get('font_key', '')
                is_title = font_key in ('title', 'heading') or 'section_header' in itype or 'title' in itype
                underflow_threshold = self._underflow_threshold(font_key, is_title)
                rewrite_allowed = self._rewrite_allowed_for_block(item, lid, text, font_key, itype)
                
                # [V10] 判断是否满足排版要求
                is_satisfied = False
                if status == 'overflow':
                    if overflow_px > 2.0:
                        issue = 'OVERFLOW'
                        if self.planner:
                            planner_feedback = self.planner.decide_feedback_path(
                                issue,
                                is_title=is_title,
                                overflow_px=overflow_px,
                                is_final_round=is_final_round,
                            )
                        decision = planner_feedback.get('decision', 'rewrite_and_tune') if planner_feedback else 'rewrite_and_tune'
                        if is_title:
                            # 标题溢出：锁字号后不再缩字号，只保留微调/接受溢出
                            print(f"   [!] Block {lid} Title Overflow {overflow_px:.1f}px -> ACCEPT MICRO-TUNE (no rewrite)")
                            is_satisfied = True
                            decision = 'micro_tune'
                        elif is_final_round or not rewrite_allowed:
                            decision = 'force_fit' if is_final_round else 'rewrite_blocked_force_fit'
                            final_style, is_satisfied, extra_actions, force_reason = self._apply_force_fit_without_font_change(
                                final_style,
                                golden_style,
                                overflow_px,
                            )
                            forced_fit = True
                            if extra_actions:
                                forced_count += 1
                                physical_actions.extend(extra_actions)
                                if is_final_round:
                                    print(f"   [Final] Block {lid} overflow {overflow_px:.1f}px -> FORCE FIT WITHOUT FONT CHANGE")
                                else:
                                    print(f"   [Guard] Block {lid} rewrite blocked, overflow {overflow_px:.1f}px -> FORCE FIT")
                            else:
                                if is_final_round:
                                    print(f"   [Final] Block {lid} minor overflow {overflow_px:.1f}px -> ACCEPT")
                                else:
                                    print(f"   [Guard] Block {lid} rewrite blocked, minor overflow {overflow_px:.1f}px -> ACCEPT")
                            final_metrics = self.tool.simulate_layout_metrics(bbox, text, final_style)
                        else:
                            action = joint_hint.get('mode') if joint_hint else (planner_feedback.get('rewrite_mode', 'shorten') if planner_feedback else 'shorten')
                            if joint_hint:
                                decision = 'column_joint_optimize'
                            print(f"   [!] Block {lid} Overflow {overflow_px:.1f}px -> SHORTEN")
                    else:
                        is_satisfied = True  # 溢出小于2px，视为满足
                        decision = 'micro_tune' if physical_actions else 'pass'
                elif status != 'overflow':
                    if final_metrics['fill_ratio'] < underflow_threshold and not is_title:
                        issue = 'UNDERFLOW'
                        current_lengthen_attempts = underflow_rewrite_attempts.get(lid, 0)
                        max_lengthen_attempts = self._max_underflow_rewrite_attempts(item, font_key)
                        if not rewrite_allowed:
                            is_satisfied = True
                            decision = 'rewrite_blocked_accept_underfill'
                        elif is_final_round or current_lengthen_attempts >= max_lengthen_attempts:
                            is_satisfied = True
                            decision = 'accept_remaining_underfill'
                        else:
                            if self.planner:
                                planner_feedback = self.planner.decide_feedback_path(
                                    issue,
                                    is_title=is_title,
                                    overflow_px=overflow_px,
                                    is_final_round=is_final_round,
                                )
                            action = planner_feedback.get('rewrite_mode', 'lengthen') if planner_feedback else 'lengthen'
                            decision = planner_feedback.get('decision', 'rewrite') if planner_feedback else 'rewrite'
                            if joint_hint:
                                action = joint_hint.get('mode', action)
                                decision = 'column_joint_optimize'
                    else:
                        is_satisfied = True
                        decision = 'pass'
                else:
                    # 非第一轮且不溢出，视为满足
                    is_satisfied = True
                    decision = 'pass'

                item['target_style'] = final_style

                if self.planner:
                    current_status = 'FORCE_FIT' if forced_fit else ('OPTIMAL' if is_satisfied else issue)
                    self.planner.update_block_state(
                        lid,
                        current_status,
                        round_index=round_i + 1,
                        reason='force_fit' if forced_fit else (status if is_satisfied else issue.lower()),
                        metadata={
                            'overflow_px': round(float(overflow_px), 3),
                            'status': status,
                            'is_title': is_title,
                            'decision': decision,
                        },
                    )
                    state_after_eval = self.planner.get_block_state(lid)
                else:
                    state_after_eval = 'FORCE_FIT' if forced_fit else ('OPTIMAL' if is_satisfied else issue)
                
                if self.planner:
                    planned_actions = planner_feedback.get('actions', []) if planner_feedback else []
                    self.planner.record_reflow_evaluation(
                        lid,
                        round_i + 1,
                        issue,
                        decision,
                        state_before,
                        state_after_eval,
                        planned_actions=planned_actions,
                        applied_actions=physical_actions,
                        metrics_before=base_metrics,
                        metrics_after=final_metrics,
                        metadata={
                            'rewrite_mode': action,
                            'overflow_px': round(float(overflow_px), 3),
                            'status': status,
                            'is_title': is_title,
                            'rewrite_allowed': rewrite_allowed,
                            'forced_fit': forced_fit,
                            'final_round': is_final_round,
                        },
                    )

                if is_satisfied and lid is not None:
                    if not self.planner:
                        stable_lids.add(lid)
                    satisfied_this_round += 1
                
                # [V10] 最后一轮不调用LLM rewrite
                if lid is not None and lid not in processed_lids and action and enable_rewrite and rewrite_allowed and not is_final_round:
                    if self.planner:
                        self.planner.update_block_state(
                            lid,
                            'REWRITE_SCHEDULED',
                            round_index=round_i + 1,
                            reason=action,
                            metadata={
                                'issue': issue,
                                'overflow_px': round(float(overflow_px), 3),
                            },
                        )
                    sibling_items = [
                        candidate for candidate in data
                        if candidate.get('logical_para_id') == lid and candidate.get('type') == item.get('type')
                    ]
                    sibling_items.sort(key=lambda x: (x['page_idx'], x['bbox'][1], x['bbox'][0]))
                    before_text = "".join(candidate.get('translated', '') for candidate in sibling_items)
                    current_chars = max(1, len(before_text.replace(' ', '').replace('\n', '')))
                    target_chars = joint_hint.get('target_chars', 0) if joint_hint else 0
                    if not target_chars:
                        ratio_target = self._rewrite_fill_target(issue, font_key)
                        current_fill_ratio = max(0.01, final_metrics.get('fill_ratio', 1.0))
                        target_chars = max(1, int(current_chars * (ratio_target / current_fill_ratio)))
                        if issue == 'UNDERFLOW' and font_key == 'body':
                            min_growth = 1.08 if self._layout_mode() == 'single_col' else 1.05
                            if current_fill_ratio < max(0.01, underflow_threshold - 0.05):
                                min_growth += 0.05
                            target_chars = max(target_chars, int(current_chars * min_growth))
                    updates = None
                    if joint_hint and self.enable_column_joint_optimization:
                        partner_lid = joint_hint.get('partner_lid')
                        if partner_lid and partner_lid not in processed_lids:
                            partner_hint = column_joint_plan.get(partner_lid)
                            partner_before = "".join(
                                candidate.get('translated', '')
                                for candidate in data
                                if candidate.get('logical_para_id') == partner_lid
                            )
                            group_specs = [
                                {
                                    'lid': lid,
                                    'mode': action,
                                    'target_chars': target_chars,
                                    'issue': issue,
                                }
                            ]
                            if partner_hint:
                                group_specs.append(
                                    {
                                        'lid': partner_lid,
                                        'mode': partner_hint.get('mode', 'lengthen'),
                                        'target_chars': partner_hint.get('target_chars', 0),
                                        'issue': 'UNDERFLOW' if partner_hint.get('mode') == 'lengthen' else 'OVERFLOW',
                                    }
                                )
                            joint_updates = self.refiner.rewrite_column_group(
                                data,
                                group_specs,
                                target_fill_ratio=0.95,
                            )
                            updates = joint_updates.get(lid)
                            if partner_lid in joint_updates:
                                partner_updates = joint_updates[partner_lid]
                                for update in partner_updates:
                                    if len(update) == 3:
                                        u_idx, u_text, style_mapping = update
                                    else:
                                        u_idx, u_text = update
                                        style_mapping = {}
                                    data[u_idx]['translated'] = u_text
                                    if style_mapping:
                                        data[u_idx]['rewrite_style_mapping'] = style_mapping
                                    if 'target_style' in data[u_idx]:
                                        del data[u_idx]['target_style']
                                processed_lids.add(partner_lid)
                                if self.planner:
                                    partner_after = "".join(
                                        data[u_idx].get('translated', '')
                                        for u_idx, *_ in partner_updates
                                    )
                                    partner_mode = partner_hint.get('mode', 'lengthen') if partner_hint else 'lengthen'
                                    self.planner.record_rewrite_result(
                                        partner_lid,
                                        round_i + 1,
                                        partner_mode,
                                        partner_before,
                                        partner_after,
                                        [u_idx for u_idx, *_ in partner_updates],
                                        metadata={
                                            'issue': 'UNDERFLOW' if partner_mode == 'lengthen' else 'OVERFLOW',
                                            'status': 'column_joint',
                                            'joint_partner_lid': lid,
                                        },
                                    )
                                    self.planner.update_block_state(
                                        partner_lid,
                                        'PENDING',
                                        round_index=round_i + 1,
                                        reason='column_joint_rewrite_applied',
                                        metadata={
                                            'rewrite_mode': partner_mode,
                                            'joint_partner_lid': lid,
                                        },
                                    )
                    if updates is None:
                        updates = self.refiner.rewrite_logical_block_with_guidance(
                            i,
                            data,
                            mode=action,
                            target_chars=target_chars,
                            current_chars=current_chars,
                            target_fill_ratio=self._rewrite_fill_target(issue, font_key),
                            current_fill_ratio=final_metrics.get('fill_ratio', 1.0),
                        )
                    if updates:
                        rewrite_count += 1  # [V9] 统计rewrite次数
                        updated_indices = []
                        for update in updates:
                            # [V8] 支持新的返回格式 (idx, text, style_mapping)
                            if len(update) == 3:
                                u_idx, u_text, style_mapping = update
                            else:
                                u_idx, u_text = update
                                style_mapping = {}
                            
                            data[u_idx]['translated'] = u_text
                            updated_indices.append(u_idx)
                            
                            # [V8 新增] 保存样式映射到 item
                            if style_mapping:
                                data[u_idx]['rewrite_style_mapping'] = style_mapping
                            
                            if 'target_style' in data[u_idx]: 
                                del data[u_idx]['target_style']
                        after_text = "".join(data[u_idx].get('translated', '') for u_idx in updated_indices)
                        if self.planner:
                            self.planner.record_rewrite_result(
                                lid,
                                round_i + 1,
                                action,
                                before_text,
                                after_text,
                                updated_indices,
                                metadata={
                                    'issue': issue,
                                    'status': status,
                                },
                            )
                            self.planner.update_block_state(
                                lid,
                                'PENDING',
                                round_index=round_i + 1,
                                reason='rewrite_applied',
                                metadata={
                                    'rewrite_mode': action,
                                    'updated_indices': updated_indices,
                                },
                            )
                        if action == 'lengthen':
                            underflow_rewrite_attempts[lid] = underflow_rewrite_attempts.get(lid, 0) + 1
                        processed_lids.add(lid)
                        needs_rewrite = True
            
            # [V10] 统计本轮满足要求的block数量
            round_stats[f'round_{round_i+1}'] = satisfied_this_round
            if self.planner:
                round_stats[f'round_{round_i+1}_planner'] = self.planner.finish_round(round_i + 1)
            print(f"\n   📊 Round {round_i+1} Summary:")
            print(f"      ⏭️  Skipped (already satisfied): {skipped_count} blocks")
            print(f"      ✅ Newly satisfied: {satisfied_this_round} blocks")
            if is_final_round:
                print(f"      🔧 Force satisfied: {forced_count} blocks")
            else:
                print(f"      🔄 Rewritten: {rewrite_count} blocks")
            
            if is_final_round:
                print(f"   ✨ Final round completed, all blocks processed.")
                break
            
            if not needs_rewrite: 
                print(f"   ✨ No more rewrites needed, stopping early.")
                break
            processed_lids.clear()

        print("[*] Drawing Final PDF...")
        output_doc = fitz.open(input_pdf)
        render_bottom_guards = self._compute_render_bottom_guards(data)
        for item_index, item in enumerate(data):
            # 再次确保 Image/Table 不被触碰
            raw_type = item.get('type', '').lower()
            if raw_type in ['image', 'table', 'figure']: continue

            itype = item.get('detected_type', raw_type).lower()
            if self._matches_preserve_type(raw_type, itype): continue
            
            text = item.get('translated', '')
            if not text: continue
            
            # [增强] 合并 target_style 和 golden_style，确保样式信息不丢失
            golden_style = item.get('golden_style', {})
            target_style = item.get('target_style', {})
            
            # 以 golden_style 为基础，用 target_style 覆盖排版相关属性
            style = {
                'size': target_style.get('size', golden_style.get('size', 9.0)),
                'line': target_style.get('line', golden_style.get('line', 1.35)),
                'char': target_style.get('char', golden_style.get('char', 0.05)),
                'font_key': target_style.get('font_key', golden_style.get('font_key', 'body')),
                # 默认样式（无粗体/斜体）
                'is_bold': False,
                'is_italic': False,
                'text_color': golden_style.get('text_color', (0, 0, 0)),
                'bg_color': golden_style.get('bg_color', None),
            }
            style = self._clamp_style_guardrails(style, golden_style or style)
            style['_render_bottom_limit'] = render_bottom_guards.get(item_index, float(item['bbox'][3]) + 3.0)
            if self._is_font_size_locked():
                style['_lock_font_size'] = True
            
            # [V8 精准样式] 优先使用重写时保存的样式映射，否则重新对齐
            rich_spans = item.get('rich_spans', [])
            source_text = item.get('text', '')
            
            # [V8 新增] 如果有重写时的样式映射，优先使用
            rewrite_style_mapping = item.get('rewrite_style_mapping', {})
            
            if rewrite_style_mapping:
                # 使用重写时已对齐的样式映射（避免重复调用 LLM）
                style_mapping = self._sanitize_style_mapping(text, rewrite_style_mapping)
            elif rich_spans and len(rich_spans) > 0:
                # 提取样式关键词
                style_keywords = self._extract_style_keywords(rich_spans)
                
                if style_keywords:
                    # 使用 LLM 进行语义对齐
                    style_mapping = self._semantic_align_styles(source_text, text, style_keywords)
                else:
                    style_mapping = {}
            else:
                style_mapping = {}
            
            if style_mapping:
                # 使用语义对齐结果进行渲染
                self.tool.draw_with_semantic_styles(
                    output_doc[item['page_idx']], 
                    item['bbox'], 
                    text,
                    style_mapping,  # {译文词: (is_bold, is_italic)}
                    style
                )
            else:
                # 无样式词，使用普通渲染
                self.tool.draw_content(output_doc[item['page_idx']], item['bbox'], text, style)

            if self.planner:
                self.planner.record_render_result(
                    item.get('logical_para_id'),
                    item_index=item_index,
                    final_style=style,
                    metrics=self.tool.simulate_layout_metrics(item['bbox'], text, style),
                    metadata={
                        'render_mode': 'semantic_style' if style_mapping else 'plain',
                        'page_idx': item.get('page_idx', 0),
                    },
                )
            
        output_doc.save(output_pdf)
        print(f"[OK] Saved: {output_pdf}")
        
        # [V9 新增] 返回统计信息
        total_blocks = len([
            d for d in data
            if d.get('type', '').lower() not in ['image', 'table', 'figure']
            and not self._matches_preserve_type(d.get('type', ''), d.get('detected_type', d.get('type', '')))
            and d.get('translated', '')
        ])
        round_stats['total_blocks'] = total_blocks
        
        print(f"\n📊 Reflow Statistics:")
        for round_num in range(iterative_rounds):
            count = round_stats.get(f'round_{round_num+1}', 0)
            percentage = (count / total_blocks * 100) if total_blocks > 0 else 0
            print(f"   Round {round_num+1}: {count}/{total_blocks} ({percentage:.1f}%)")

        if self.planner:
            round_stats['planner_action_stats'] = dict(self.planner.action_counter)
        
        return round_stats
