import statistics
import re
import time
import fitz
import math
from collections import Counter
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from tools.api_client import APIClient
from tools.pdf_reflow_tool import PDFReflowTool, sanitize_inline_math_text
from tools.source_text_recovery import recover_low_information_items

class ContentMasker:
    """[组件] 内容保护器 + [V10] LaTeX转Unicode"""
    def __init__(self):
        self.formula_pattern = re.compile(r'(?<!\\)\$(?:\\.|[^$])+(?<!\\)\$')
        self.placeholder_prefix = "MATH_MASK_"
        
        # [V10] LaTeX到Unicode映射
        self.latex_to_unicode = {
            r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
            r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Phi': 'Φ',
            r'\Psi': 'Ψ', r'\Omega': 'Ω',
            r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
            r'\epsilon': 'ε', r'\theta': 'θ', r'\lambda': 'λ', r'\mu': 'μ',
            r'\nu': 'ν', r'\pi': 'π', r'\rho': 'ρ', r'\sigma': 'σ',
            r'\tau': 'τ', r'\phi': 'φ', r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
            r'\times': '×', r'\leq': '≤', r'\geq': '≥', r'\neq': '≠',
            r'\in': '∈', r'\subset': '⊂', r'\cup': '∪', r'\cap': '∩',
            r'\forall': '∀', r'\exists': '∃', r'\infty': '∞',
            r'\sum': '∑', r'\prod': '∏', r'\int': '∫',
            r'\coloneqq': '≔', r'\colon': ':', r'\to': '→', r'\rightarrow': '→',
            r'\mathbb{N}': 'ℕ', r'\mathbb{Z}': 'ℤ', r'\mathbb{Q}': 'ℚ',
            r'\mathbb{R}': 'ℝ', r'\mathbb{C}': 'ℂ',
            r'\mid': '|', r'\ldots': '…', r'\geqslant': '≥', r'\leqslant': '≤',
        }
    
    def _convert_latex_to_unicode(self, text: str) -> str:
        """[V11修复] 将LaTeX公式转换为Unicode，正确处理花括号"""
        if not text:
            return text
        
        result = text
        
        # [关键修复] 先处理转义的花括号（必须最先）
        result = result.replace(r'\{', '{')
        result = result.replace(r'\}', '}')
        
        # 修复被空格拆分的LaTeX命令
        result = re.sub(r'\\col\s+oneqq', r'\\coloneqq', result)
        result = re.sub(r'\\col\s+on', r'\\colon', result)
        result = re.sub(r'\\math\s*bb', r'\\mathbb', result)
        
        # 替换LaTeX命令（按长度从长到短）
        for latex, unicode_char in sorted(self.latex_to_unicode.items(), key=lambda x: len(x[0]), reverse=True):
            result = result.replace(latex, unicode_char)
        
        # 移除数学模式标记
        result = result.replace('$', '')
        
        # 清理LaTeX命令（保留花括号内容）
        result = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', result)  # \command{text} -> text
        result = re.sub(r'\\[a-zA-Z]+', '', result)  # 移除其他命令
        
        # 清理多余的空格
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()
    
    def clean_html_tags(self, text: str) -> str:
        """清理HTML标签，保留内容但移除标签"""
        if not text:
            return text
        
        # 处理上标 <sup>text</sup> 或 <sup>text</sup>
        text = re.sub(r'<sup>([^<]+)</sup>', r'^\1', text, flags=re.IGNORECASE)
        
        # 处理下标 <sub>text</sub>
        text = re.sub(r'<sub>([^<]+)</sub>', r'_\1', text, flags=re.IGNORECASE)
        
        # 处理斜体 <i>text</i> 或 <em>text</em> - 保留内容
        text = re.sub(r'<(?:i|em)>([^<]+)</(?:i|em)>', r'\1', text, flags=re.IGNORECASE)
        
        # 处理粗体 <b>text</b> 或 <strong>text</strong> - 保留内容
        text = re.sub(r'<(?:b|strong)>([^<]+)</(?:b|strong)>', r'\1', text, flags=re.IGNORECASE)
        
        # 处理其他常见标签 - 直接移除标签保留内容
        text = re.sub(r'</?(?:span|div|p)[^>]*>', '', text, flags=re.IGNORECASE)
        
        # 清理可能残留的其他HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        return text

    def mask(self, text: str) -> Tuple[str, Dict[str, str]]:
        # 先清理HTML标签
        text = self.clean_html_tags(text)
        
        matches = list(self.formula_pattern.finditer(text))
        masked_text = text
        mapping = {}
        for i, m in enumerate(reversed(matches)):
            original = m.group(0)
            key = f"[{self.placeholder_prefix}{len(matches)-1-i}]"
            mapping[key] = original
            start, end = m.span()
            masked_text = masked_text[:start] + key + masked_text[end:]
        return masked_text, mapping

    def unmask(self, translated_text: str, mapping: Dict[str, str]) -> str:
        """[V21图片优先] unmask时保持LaTeX原样，交给PDF渲染器用图片"""
        final_text = translated_text
        
        # 1. 精确匹配替换（保持LaTeX原样）
        for key, original in mapping.items():
            if key in final_text:
                final_text = final_text.replace(key, original)
        
        # 2. 宽松匹配替换（保持LaTeX原样）
        def replace_loose(match):
            idx = match.group(1)
            key = f"[{self.placeholder_prefix}{idx}]"
            return mapping.get(key, match.group(0))
        
        # 支持多种占位符变体:
        # [MATH_MASK_0], [ MATH_MASK_0 ], [MATH MASK 0], [MATHASK_0], MATH_MASK_0
        patterns = [
            r'\[\s*' + self.placeholder_prefix + r'(\d+)\s*\]',  # [MATH_MASK_0] 或 [ MATH_MASK_0 ]
            r'\[\s*MATH\s+MASK\s*_?\s*(\d+)\s*\]',  # [MATH MASK 0] 或 [MATH MASK_0]
            r'\[\s*MATHASK\s*_?\s*(\d+)\s*\]',  # [MATHASK_0] (缺少M)
            r'(?<!\[)\s*' + self.placeholder_prefix + r'(\d+)(?!\])',  # MATH_MASK_0 (缺少方括号)
        ]
        
        for pattern in patterns:
            try:
                loose_pattern = re.compile(pattern)
                final_text = loose_pattern.sub(replace_loose, final_text)
            except:
                pass  # 忽略正则表达式错误
        
        # [V16] 终极防护：强制清理任何残留的占位符
        final_text = re.sub(r'\[MATH_MASK_\d+\]', ' ', final_text)
        final_text = re.sub(r'\[\s*MATH\s*MASK\s*_?\s*\d+\s*\]', ' ', final_text, flags=re.IGNORECASE)
        return sanitize_inline_math_text(final_text)

class TranslationAgent:
    def __init__(self, api_client: APIClient, target_lang='zh', planner=None):
        self.client = api_client
        self.target_lang = target_lang
        self.planner = planner
        self.masker = ContentMasker()
        
        # [关键] 明确不处理的类型 (保留原文及坐标)
        self.IGNORE_TYPES = {
            'image', 'table', 'line', 'rect', 'curve', 'equation', 'code',
            'author', 'affiliation', 'reference',
            'aside_text', 'page_number',
        }
        
        # 必须翻译的类型
        self.TARGET_TYPES = {
            'text', 'title', 'section_header', 'page_header', 'header', 'footer',
            'caption', 'figure_caption', 'table_caption', 'footnote', 'abstract'
        }
        
        # 需要翻译但不允许扩写的类型
        self.NO_EXPANSION_TYPES = {
            'title', 'section_header', 'page_header', 'header', 'footer', 'caption', 
            'figure_caption', 'table_caption', 'footnote'
        }
        
        # 需要跳过翻译的文本类型 (如作者/参考文献，通常保留原文)
        self.SKIP_TEXT_TYPES = {
            'author', 'affiliation', 'reference',
            'aside_text', 'page_number',
        }
        
        # [V5 新增] 常见学术术语直接翻译映射（避免 API 出错）
        self.DIRECT_TRANSLATION_MAP = {
            'ABSTRACT': '摘要',
            'Abstract': '摘要',
            'INTRODUCTION': '引言',
            'Introduction': '引言',
            'CONCLUSION': '结论',
            'Conclusion': '结论',
            'CONCLUSIONS': '结论',
            'Conclusions': '结论',
            'REFERENCES': '参考文献',
            'References': '参考文献',
            'ACKNOWLEDGMENTS': '致谢',
            'Acknowledgments': '致谢',
            'ACKNOWLEDGEMENTS': '致谢',
            'Acknowledgements': '致谢',
            'RELATED WORK': '相关工作',
            'Related Work': '相关工作',
            'METHODOLOGY': '方法论',
            'Methodology': '方法论',
            'METHOD': '方法',
            'Method': '方法',
            'METHODS': '方法',
            'Methods': '方法',
            'RESULTS': '结果',
            'Results': '结果',
            'DISCUSSION': '讨论',
            'Discussion': '讨论',
            'EXPERIMENTS': '实验',
            'Experiments': '实验',
            'APPENDIX': '附录',
            'Appendix': '附录',
            'SUPPLEMENTARY': '补充材料',
            'Supplementary': '补充材料',
            'BACKGROUND': '背景',
            'Background': '背景',
            'OVERVIEW': '概述',
            'Overview': '概述',
            'EVALUATION': '评估',
            'Evaluation': '评估',
            'ANALYSIS': '分析',
            'Analysis': '分析',
            'LIMITATIONS': '局限性',
            'Limitations': '局限性',
            'FUTURE WORK': '未来工作',
            'Future Work': '未来工作',
        }
        
        self.layout_mode = "single_col"
        self.global_style_config = {}
        self._font_size_cache = {}
        self._split_layout_tool = None

    REFERENCE_HEADINGS = {
        'references', 'reference', 'bibliography', 'works cited'
    }

    def _get_layout_tool(self):
        if self._split_layout_tool is None:
            self._split_layout_tool = PDFReflowTool(lang=self.target_lang)
        return self._split_layout_tool
        print(f"🌍 [TranslationAgent] Mode: Robust Type Classification & Strict Coordinates")

    def _get_original_metrics(self, item: Dict, doc: fitz.Document) -> float:
        """从 PDF 精确提取 BBox 内的原始字号"""
        idx = item.get('logical_para_id', -1)
        if idx in self._font_size_cache: return self._font_size_cache[idx]
        try:
            page = doc[item['page_idx']]
            rect = fitz.Rect(item['bbox'])
            text_blocks = page.get_text("dict", clip=rect)["blocks"]
            sizes = []
            for b in text_blocks:
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        if s["size"] > 0 and len(s["text"].strip()) > 1:
                            sizes.append(s["size"])
            size = statistics.median(sizes) if sizes else 9.0
            self._font_size_cache[idx] = size
            return size
        except:
            return 9.0

    def _sample_background_color(self, page: fitz.Page, rect: fitz.Rect) -> Tuple[float, float, float] | None:
        try:
            sample_rect = fitz.Rect(
                max(page.rect.x0, rect.x0 - 1.5),
                max(page.rect.y0, rect.y0 - 1.5),
                min(page.rect.x1, rect.x1 + 1.5),
                min(page.rect.y1, rect.y1 + 1.5),
            )
            pix = page.get_pixmap(clip=sample_rect, dpi=72, alpha=False)
            if pix.width < 3 or pix.height < 3:
                return None

            border = max(1, min(3, min(pix.width, pix.height) // 8))
            colors = Counter()
            for y in range(pix.height):
                for x in range(pix.width):
                    if border <= x < pix.width - border and border <= y < pix.height - border:
                        continue
                    sample = pix.pixel(x, y)
                    if len(sample) < 3:
                        continue
                    rgb = tuple(int(round(channel / 8.0) * 8) for channel in sample[:3])
                    colors[rgb] += 1

            if not colors:
                return None

            rgb = colors.most_common(1)[0][0]
            normalized = tuple(channel / 255.0 for channel in rgb)
            if all(channel > 0.98 for channel in normalized):
                return None
            return normalized
        except Exception:
            return None

    def _score_background_fill_candidate(
        self,
        rect: fitz.Rect,
        candidate_rect: fitz.Rect,
        overlap_ratio: float,
    ) -> Tuple[int, float, float, float]:
        expanded_rect = fitz.Rect(rect.x0 - 0.2, rect.y0 - 0.2, rect.x1 + 0.2, rect.y1 + 0.2)
        contains_rect = candidate_rect.contains(expanded_rect) or overlap_ratio >= 0.995
        rect_area = max(1.0, rect.get_area())
        candidate_area = max(candidate_rect.get_area(), 1.0)
        extra_area = max(0.0, (candidate_area / rect_area) - 1.0)
        specificity = 1.0 / (1.0 + extra_area)
        return (
            1 if contains_rect else 0,
            round(overlap_ratio, 6),
            round(specificity, 6),
            -round(extra_area, 6),
        )

    def _extract_explicit_background_color(self, page: fitz.Page, rect: fitz.Rect) -> Tuple[float, float, float] | None:
        rect_area = max(1.0, rect.get_area())
        best_fill = None
        best_score = None
        for drawing in page.get_drawings():
            fill = drawing.get("fill")
            draw_rect = drawing.get("rect")
            if not fill or not draw_rect:
                continue
            d_rect = fitz.Rect(draw_rect)
            if not d_rect.intersects(rect):
                continue
            inter = d_rect & rect
            inter_area = inter.get_area() if inter else 0.0
            if inter_area <= 0:
                continue
            overlap_ratio = inter_area / rect_area
            normalized_fill = tuple(float(c) for c in fill[:3])
            if all(c >= 0.97 for c in normalized_fill):
                continue
            candidate_area = max(d_rect.get_area(), 1.0)
            area_ratio = candidate_area / rect_area
            # 只接受贴合文本框的色块，避免把整幅图/整页的底色误当作文字背景。
            if area_ratio > 2.6:
                continue
            if overlap_ratio < 0.86 and area_ratio > 1.6:
                continue
            score = self._score_background_fill_candidate(rect, d_rect, overlap_ratio)
            if overlap_ratio >= 0.72 and (best_score is None or score > best_score):
                best_fill = normalized_fill
                best_score = score
        return best_fill

    def _is_reference_heading(self, text: str) -> bool:
        normalized = " ".join((text or "").strip().lower().split())
        return normalized in self.REFERENCE_HEADINGS

    def _looks_like_reference_entry(self, text: str) -> bool:
        compact = " ".join((text or "").split())
        if len(compact) < 24:
            return False
        if re.search(r'(^|\s)\[[^\]]+\]\s+', compact):
            return True
        if re.search(r'(^|\s)\d+\.\s+[A-Z]', compact):
            return True
        if re.search(r'Email address:|https?://|@', compact, re.IGNORECASE):
            return True
        if re.search(r'\b(arXiv|doi|vol\.|pp\.|Proc\.|Journal|Press|University|Univ\.|Springer|Cambridge)\b', compact, re.IGNORECASE) and re.search(r'\b(19|20)\d{2}\b', compact):
            return True
        if re.search(r'[A-Z][a-zA-Z\'`\-]+,\s*(?:[A-Z]\.){1,3}', compact) and re.search(r'\b(19|20)\d{2}\b', compact):
            return True
        return False

    def _looks_like_reference_entry_strong(self, text: str) -> bool:
        compact = " ".join((text or "").split())
        if len(compact) < 32:
            return False
        year_hits = len(re.findall(r'\b(?:19|20)\d{2}[a-z]?\b', compact))
        author_hits = len(re.findall(r'[A-Z][a-zA-Z\'`\-]+,\s*(?:[A-Z]\.){1,3}', compact))
        journal_like = re.search(
            r'\b(?:ApJ|ApJS|A&A|MNRAS|AJ|arXiv|doi|Proc\.|Proceedings|Journal|Conference|Vol\.|pp\.)\b',
            compact,
            re.IGNORECASE,
        )
        etal = re.search(r'\bet\s+al\.?\b', compact, re.IGNORECASE)
        return year_hits >= 1 and (author_hits >= 2 or etal) and bool(journal_like or year_hits >= 2)

    def _looks_like_reference_fragment(self, text: str) -> bool:
        compact = " ".join((text or "").split())
        if len(compact) < 20:
            return False
        if re.search(r'Email address:|https?://|@', compact, re.IGNORECASE):
            return False

        year_hits = len(re.findall(r'\b(?:19|20)\d{2}[a-z]?\b', compact))
        author_hits = len(re.findall(r'[A-Z][a-zA-Z\'`\-]+,\s*(?:[A-Z]\.){1,3}', compact))
        initial_runs = len(re.findall(r'(?:^|[\s,;&])(?:[A-Z]\.\s*){1,3}[A-Z][a-zA-Z\'`\-]+', compact))
        comma_chunks = compact.count(',')
        ampersand = '&' in compact or ' and ' in compact.lower()
        journal_like = re.search(
            r'\b(?:ApJ|ApJS|A&A|MNRAS|AJ|Science|Nature|arXiv|doi|Proc\.|Proceedings|Journal|Conference|Vol\.|pp\.)\b',
            compact,
            re.IGNORECASE,
        )
        page_like = re.search(r'\b\d{1,4}\b(?:\s*[-–]\s*\d{1,4})?\b', compact)

        if year_hits >= 1 and (author_hits >= 1 or initial_runs >= 2):
            return True
        if year_hits >= 1 and journal_like and (comma_chunks >= 3 or ampersand):
            return True
        if journal_like and (author_hits >= 1 or initial_runs >= 2) and page_like:
            return True
        return False

    def _annotate_reference_sections(self, groups: Dict, sorted_lids: List[int], body_base_size: float, doc: fitz.Document, data: List[Dict] | None = None) -> None:
        ordered_entries = []
        for lid in sorted_lids:
            group = groups[lid]
            group.sort(key=lambda x: (x['item']['page_idx'], x['item']['bbox'][1]))
            first_item = group[0]['item']
            raw_text = first_item.get('context') or " ".join(g['item'].get('text', '') for g in group)
            corrected_type = self._correct_element_type_visually(first_item, body_base_size, doc)
            ordered_entries.append({
                'page_idx': first_item['page_idx'],
                'y0': first_item['bbox'][1],
                'raw_text': raw_text,
                'corrected_type': corrected_type,
                'raw_type': first_item.get('type', '').lower(),
                'group': group,
            })

        if data:
            for item in data:
                raw_text = (item.get('context') or item.get('text') or '').strip()
                raw_type = item.get('type', '').lower()
                if not raw_text:
                    continue
                if 'reference' not in raw_type and not self._is_reference_heading(raw_text):
                    continue
                ordered_entries.append({
                    'page_idx': item.get('page_idx', 0),
                    'y0': item.get('bbox', [0, 0, 0, 0])[1],
                    'raw_text': raw_text,
                    'corrected_type': 'reference' if 'reference' in raw_type else self._correct_element_type_visually(item, body_base_size, doc),
                    'raw_type': raw_type,
                    'group': None,
                })

        ordered_entries.sort(key=lambda entry: (entry['page_idx'], entry['y0']))
        late_page_threshold = max(0, int(doc.page_count * 0.35))
        in_reference_section = False
        for entry in ordered_entries:
            group = entry['group']
            raw_text = entry['raw_text']
            stripped_text = raw_text.strip()
            corrected_type = entry['corrected_type']
            raw_type = entry['raw_type']
            page_idx = entry['page_idx']
            strong_reference = self._looks_like_reference_entry_strong(raw_text)
            weak_reference = self._looks_like_reference_entry(raw_text)
            fragment_reference = self._looks_like_reference_fragment(raw_text)

            preserve = False
            if self._is_reference_heading(stripped_text):
                in_reference_section = True
                preserve = True
            elif corrected_type == 'reference' or 'reference' in raw_type:
                preserve = True
                if page_idx >= late_page_threshold and (
                    strong_reference or weak_reference or len(stripped_text) <= 32
                ):
                    in_reference_section = True
            elif strong_reference and (
                in_reference_section
                or page_idx >= late_page_threshold
                or corrected_type in {'footnote', 'footer', 'header'}
            ):
                preserve = True
                in_reference_section = True
            elif in_reference_section and (weak_reference or fragment_reference):
                preserve = True
            elif in_reference_section and corrected_type in {'text', 'footnote'} and (
                fragment_reference
                or (
                    len(stripped_text) >= 20
                    and re.search(r'\b(?:19|20)\d{2}[a-z]?\b', stripped_text)
                    and (',' in stripped_text or '&' in stripped_text)
                )
            ):
                preserve = True

            if preserve and group:
                for g in group:
                    g['item']['_preserve_reference'] = True

    def _should_preserve_original_group(self, corrected_type: str, raw_text: str, group: List[Dict]) -> bool:
        if self._is_reference_heading(raw_text):
            return True
        if corrected_type == 'reference':
            return True
        return any(g['item'].get('_preserve_reference') for g in group)

    def _extract_style_info(self, item: Dict, doc: fitz.Document) -> Dict:
        """
        [新增] 从 PDF 精确提取 BBox 内的样式信息
        包括：字体粗细、斜体、颜色、背景色、原始字号等
        """
        style_info = {
            'is_bold': False,
            'is_italic': False,
            'text_color': (0, 0, 0),  # 默认黑色 RGB
            'bg_color': None,  # None 表示透明/白色
            'bg_source': None,
            'font_flags': 0,
            'original_font': None,
            'original_size': None  # [V7 新增] 原始字号
        }
        
        try:
            page = doc[item['page_idx']]
            rect = fitz.Rect(item['bbox'])
            
            # 1. 提取文本样式
            text_blocks = page.get_text("dict", clip=rect)["blocks"]
            
            font_flags_list = []
            colors_list = []
            fonts_list = []
            sizes_list = []  # [V7 新增] 收集字号
            
            for b in text_blocks:
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        if len(s.get("text", "").strip()) < 1:
                            continue
                        
                        # 字体标志位: bit 0=superscript, bit 1=italic, bit 2=serifed, bit 3=monospaced, bit 4=bold
                        flags = s.get("flags", 0)
                        font_flags_list.append(flags)
                        
                        # 颜色 (PyMuPDF 返回整数，需要转换为 RGB)
                        color_int = s.get("color", 0)
                        if isinstance(color_int, int):
                            # 转换为 RGB 元组 (0-1 范围)
                            r = ((color_int >> 16) & 0xFF) / 255.0
                            g = ((color_int >> 8) & 0xFF) / 255.0
                            b_val = (color_int & 0xFF) / 255.0
                            colors_list.append((r, g, b_val))
                        
                        # 原始字体名
                        font_name = s.get("font", "")
                        if font_name:
                            fonts_list.append(font_name)
                        
                        # [V7 新增] 原始字号
                        font_size = s.get("size", 0)
                        if font_size > 0:
                            sizes_list.append(font_size)
            
            # 2. 统计主要样式（使用众数）
            if font_flags_list:
                # 取最常见的 flags
                most_common_flags = Counter(font_flags_list).most_common(1)[0][0]
                style_info['font_flags'] = most_common_flags
                # bit 4 = bold, bit 1 = italic
                style_info['is_bold'] = bool(most_common_flags & (1 << 4))
                style_info['is_italic'] = bool(most_common_flags & (1 << 1))
            
            if colors_list:
                # 取最常见的颜色
                # 将浮点数四舍五入以便比较
                rounded_colors = [tuple(round(c, 2) for c in col) for col in colors_list]
                most_common_color = Counter(rounded_colors).most_common(1)[0][0]
                style_info['text_color'] = most_common_color
            
            if fonts_list:
                style_info['original_font'] = Counter(fonts_list).most_common(1)[0][0]
            
            # [V7 新增] 提取最大字号（用于标题）
            if sizes_list:
                style_info['original_size'] = max(sizes_list)
            
            # 3. 提取背景色 (检测该区域的图像/矩形填充)
            # 方法：检查页面上覆盖该区域的矩形/路径的填充色
            explicit_bg = self._extract_explicit_background_color(page, rect)
            if explicit_bg is not None:
                style_info['bg_color'] = explicit_bg
                style_info['bg_source'] = 'vector_fill'
                    
        except Exception as e:
            pass  # 保持默认值
            
        return style_info

    def _detect_layout_mode(self, body_items: List[Dict], page_width: float) -> str:
        if not body_items: return "single_col"
        center_x = page_width / 2
        gutter = page_width * 0.05
        crossing_count = sum(1 for i in body_items if i['bbox'][0] < center_x - gutter and i['bbox'][2] > center_x + gutter)
        penetration_rate = crossing_count / len(body_items)
        widths = [i['bbox'][2] - i['bbox'][0] for i in body_items]
        avg_width_ratio = statistics.mean(widths) / page_width

        # 稍微放宽双栏判定标准
        if penetration_rate > 0.15: return "single_col"
        if avg_width_ratio < 0.55: return "double_col" # 从 0.48 放宽到 0.55
        return "single_col"

    def _estimate_single_col_body_targets(self, body_items: List[Dict], page_width: float, avg_src_density: float) -> Tuple[float, float]:
        widths = [
            max(1.0, float(item['bbox'][2]) - float(item['bbox'][0]))
            for item in body_items
            if len(item.get('bbox', [])) == 4
        ]
        avg_width_ratio = (statistics.mean(widths) / page_width) if widths and page_width > 0 else 0.85
        math_like_count = 0
        for item in body_items:
            text = item.get('text', '') or ''
            if '$' in text or re.search(r'\\[A-Za-z]+', text):
                math_like_count += 1
        math_like_ratio = math_like_count / max(1, len(body_items))

        target_line = 1.5
        target_natural_fill = 0.91

        if avg_width_ratio < 0.78:
            target_line -= 0.05
        if avg_src_density >= 80:
            target_line -= 0.03
            target_natural_fill += 0.01
        if math_like_ratio >= 0.10:
            target_line -= 0.02
            target_natural_fill -= 0.01
        if avg_width_ratio >= 0.84 and avg_src_density <= 70 and math_like_ratio <= 0.05:
            target_line += 0.03
            target_natural_fill += 0.01

        target_line = round(max(1.4, min(1.55, target_line)) * 20) / 20
        target_natural_fill = max(0.88, min(0.93, target_natural_fill))
        return target_line, target_natural_fill

    def _clean_hallucinations(self, text: str) -> str:
        """[V9 终极增强] 清洗所有非译文内容"""
        if not text:
            return ""
        
        original_text = text
        
        # 1. [优先检测] 提示词泄露 - 在清理前检测原始文本
        # 注意：不要包含可能出现在正常翻译中的内容！
        prompt_leak_indicators = [
            # 明确的元指令（英文全大写，不可能出现在译文中）
            'CRITICAL RULES', 'ABSOLUTE RULES', 'MUST FOLLOW', 'VIOLATION WILL FAIL',
            'PRESERVE ALL', 'Output ONLY', 'NO explanations', 'NO chatty',
            'LENGTH CONTROL', 'EXACT format required', 'REQUIREMENTS:',
            # 占位符说明文字（不是占位符本身！）
            'MATH_MASK_n', 'placeholders EXACTLY', 'preserve all.*placeholders',
            # LLM拒绝输出
            'I\'m sorry', 'I cannot', 'I apologize', 'as an AI', 'as a language model',
            '抱歉，我无法', '对不起，我不能', '作为一个AI', '作为语言模型',
            # 明显的重复指令
            'Here is the translation', 'The translation is',
        ]
        
        for indicator in prompt_leak_indicators:
            if indicator in text:
                print(f"    ❌ [Hallucination] Prompt leak detected: '{indicator}'")
                return ""
        
        # 2. 移除常见的翻译前缀/后缀（确认不是泄露后才清理）
        prefixes = [
            r'^(原文|Original|Translation|译文|Trans|Answer|Output|Result|翻译结果|翻译为|译为)[:：\s]*',
            r'^(Here is|Here\'s|This is|以下是|根据|按照|依据).*?[:：]\s*',
            r'^["\'\u201c\u201d\u2018\u2019]+',  # 开头的各种引号
        ]
        for pattern in prefixes:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE).strip()
        
        # 3. 移除结尾的引号和无关符号
        text = text.strip('"\'\u201c\u201d\u2018\u2019`\n\r ')
        
        # 4. 移除包含元指令的句子（仅移除明显的说明性句子）
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            # 跳过纯符号行
            if re.match(r'^[-=*#]+$', line.strip()):
                continue
            # 跳过明显的说明性前缀（区分大小写）
            if re.match(r'^(Note:|Explanation:|说明：|备注：)', line.strip()):
                continue
            clean_lines.append(line)
        
        text = '\n'.join(clean_lines).strip()
        
        # 5. 最终验证：如果清理导致文本几乎全部丢失，返回原文
        if len(text) < 5 and len(original_text) > 20:
            print(f"    ⚠️ [Cleaning Warning] Over-cleaning detected: {len(original_text)} → {len(text)}, using original")
            return original_text.strip()
        
        # 如果清理后长度变化超过50%，发出警告但仍返回清理后的文本
        if len(text) < len(original_text) * 0.5 and len(text) > 0:
            print(f"    ⚠️ [Cleaning Warning] Text shortened: {len(original_text)} → {len(text)}")
        
        return text

    def _contains_spurious_meta_output(self, source_text: str, translated_text: str) -> bool:
        src = re.sub(r'\s+', ' ', (source_text or '')).strip()
        out = re.sub(r'\s+', ' ', (translated_text or '')).strip()
        if not out:
            return False
        if not src:
            meta_markers = [
                r'请提供需要翻译',
                r'仅输出翻译内容',
                r'上一次的尝试',
                r'本次尝试',
                r'之前的尝试',
                r'翻译准确且自然',
                r'不添加其他文字',
                r'输出格式',
            ]
            return any(re.search(pattern, out) for pattern in meta_markers) or len(out) > 32

        meta_markers = [
            r'请提供需要翻译',
            r'仅输出翻译内容',
            r'上一次的尝试',
            r'本次尝试',
            r'之前的尝试',
            r'翻译准确且自然',
            r'不添加其他文字',
            r'输出格式',
            r'规则：',
            r'要求：',
        ]
        marker_hits = sum(1 for pattern in meta_markers if re.search(pattern, out))
        if marker_hits == 0:
            return False

        source_promptish = re.search(
            r'\b(?:prompt|instruction|instructions|rules?|requirements?|output only|format|previous attempt|this attempt|please provide|assistant|system)\b',
            src,
            re.IGNORECASE,
        )
        if re.search(r'请提供需要翻译', out) and not re.search(r'\bplease provide\b', src, re.IGNORECASE):
            return True
        return marker_hits >= 2 and not source_promptish

    def _count_cjk_chars(self, text: str) -> int:
        return len(re.findall(r'[\u4e00-\u9fff]', text or ""))

    def _count_latin_words(self, text: str) -> int:
        return len(re.findall(r'\b[A-Za-z]{3,}\b', text or ""))

    def _strip_formula_like_content(self, text: str) -> str:
        stripped = re.sub(r'\[MATH_MASK_\d+\]', ' ', text or '')
        stripped = re.sub(r'(?<!\\)\$(?:\\.|[^$])+(?<!\\)\$', ' ', stripped)
        return re.sub(r'\s+', ' ', stripped).strip()

    def _content_signal_score(self, text: str) -> int:
        plain = self._strip_formula_like_content(text)
        if not plain:
            return 0
        cjk = self._count_cjk_chars(plain)
        latin = len(re.findall(r'\b[A-Za-z]{2,}\b', plain))
        digits = len(re.findall(r'\d', plain))
        punctuation = len(re.findall(r'[，。；：,.!?;:]', plain))
        return cjk + latin * 4 + digits + punctuation

    def _translation_looks_unfinished(self, source_text: str, translated_text: str) -> bool:
        src = re.sub(r'\s+', ' ', (source_text or '')).strip()
        out = re.sub(r'\s+', ' ', (translated_text or '')).strip()
        if not src or not out:
            return True

        src_latin = self._count_latin_words(src)
        if src_latin < 4:
            return False

        out_latin = self._count_latin_words(out)
        out_cjk = self._count_cjk_chars(out)
        similarity = SequenceMatcher(None, src[:2500], out[:2500]).ratio()

        if out == src:
            return True
        if similarity > 0.88 and out_cjk < max(8, src_latin // 3):
            return True
        if out_cjk < max(8, src_latin // 3) and out_latin >= max(8, int(src_latin * 0.55)):
            return True
        if re.search(r'(?:\b[A-Za-z]{3,}\b[\s,;:()\-]*){12,}', out) and out_cjk < max(18, src_latin):
            return True

        src_signal = self._content_signal_score(src)
        out_signal = self._content_signal_score(out)
        src_plain = self._strip_formula_like_content(src)
        out_plain = self._strip_formula_like_content(out)
        structural_source = re.search(
            r'\b(?:theorem|lemma|proposition|corollary|definition|remark|example|proof|claim|construction|algorithm|case|step|suppose|assume|let|there\s+(?:is|exists)|such\s+that|where|then|for\s+all)\b',
            src_plain,
            re.IGNORECASE,
        )
        heading_only_output = re.match(
            r'^\s*(?:定理|引理|命题|推论|定义|注记|备注|例|证明|构造|算法|步骤|情形|情况)[^。；：:!?]*[。；：:!?]?\s*$',
            out_plain,
        )
        if src_latin >= 8 and src_signal >= 36:
            if out_signal <= max(10, int(src_signal * 0.18)) and (
                structural_source or heading_only_output or len(out_plain) <= 12
            ):
                return True
            if structural_source and heading_only_output and out_signal < max(14, int(src_signal * 0.28)):
                return True
        if structural_source and heading_only_output and src_signal >= 18:
            if out_signal <= max(8, int(src_signal * 0.35)) or len(out_plain) <= 12:
                return True
        return False

    def _strict_translate_masked_text(self, masked_text: str, mask_count: int, current_text: str = "") -> str:
        previous_output = current_text.strip()
        prompt = (
            "Translate the following academic paper text into Chinese.\n"
            "You must translate all English prose into natural academic Chinese.\n"
            "Do not leave English sentences untranslated.\n"
            "Keep citations, theorem numbers, section numbers, symbols, and notation unchanged.\n"
        )
        if mask_count > 0:
            prompt += f"Preserve all {mask_count} [MATH_MASK_X] placeholders exactly and keep their order unchanged.\n"
        if previous_output:
            prompt += (
                "\nThe previous attempt left too much English unchanged. "
                "Fix that in this attempt.\n"
                f"Previous attempt:\n{previous_output[:1200]}\n"
            )
        prompt += f"\nText:\n{masked_text}\n\nOutput only the Chinese translation."
        return self.client.chat_completion([{"role": "user", "content": prompt}], temperature=0.1)

    def _sentencewise_translate_masked_text(self, masked_text: str) -> str:
        pieces = [p.strip() for p in re.split(r'(?<=[\.\?!;:])\s+', masked_text) if p.strip()]
        if len(pieces) <= 1:
            return ""

        translated_parts = []
        for piece in pieces:
            piece_mask_count = len(re.findall(r'\[MATH_MASK_\d+\]', piece))
            translated_piece = self._strict_translate_masked_text(piece, piece_mask_count)
            translated_piece = self._clean_hallucinations(translated_piece)
            if not translated_piece:
                return ""
            translated_parts.append(translated_piece.strip())
        return " ".join(translated_parts).strip()

    def _translate_plain_text_strict(self, text: str) -> str:
        if not text.strip() or self._count_latin_words(text) < 3:
            return text
        prompt = (
            "Translate the following academic paper prose into Chinese.\n"
            "Translate every English sentence into natural academic Chinese.\n"
            "Do not leave any full English sentence untranslated.\n"
            "Keep citation indices, numbers, theorem labels, symbols, and abbreviations unchanged when appropriate.\n"
            "Output only the Chinese translation.\n\n"
            f"Text:\n{text}"
        )
        translated = self.client.chat_completion([{"role": "user", "content": prompt}], temperature=0.05)
        translated = self._clean_hallucinations(translated)
        if translated and not self._translation_looks_unfinished(text, translated):
            return translated

        retry_prompt = (
            "The previous attempt left too much English unchanged.\n"
            "Translate ALL English prose into Chinese this time.\n"
            "Keep symbols, formulas, citations, and notation unchanged.\n"
            "Output only the Chinese translation.\n\n"
            f"Text:\n{text}\n\n"
            f"Previous attempt:\n{translated[:1200] if translated else ''}"
        )
        repaired = self.client.chat_completion([{"role": "user", "content": retry_prompt}], temperature=0.0)
        repaired = self._clean_hallucinations(repaired)
        return repaired or translated

    def _translate_text_preserving_formulas(self, raw_text: str) -> str:
        parts = re.split(r'((?<!\\)\$(?:\\.|[^$])+(?<!\\)\$)', raw_text or "")
        rebuilt = []
        for part in parts:
            if not part:
                continue
            if self.masker.formula_pattern.fullmatch(part):
                rebuilt.append(part)
                continue
            paragraphs = part.split('\n')
            translated_paragraphs = []
            for paragraph in paragraphs:
                stripped = paragraph.strip()
                if not stripped:
                    translated_paragraphs.append(paragraph)
                    continue
                sentences = [piece.strip() for piece in re.split(r'(?<=[\.\?!;:])\s+', stripped) if piece.strip()]
                translated_sentences = []
                for sentence in sentences:
                    translated_sentence = self._translate_plain_text_strict(sentence)
                    translated_sentences.append(translated_sentence if translated_sentence else sentence)
                translated_paragraphs.append(" ".join(translated_sentences).strip())
            rebuilt.append("\n".join(translated_paragraphs).strip())
        return "".join(rebuilt).strip()

    def _repair_untranslated_output(self, raw_text: str, masked_text: str, translated_text: str, mask_count: int) -> str:
        candidate = translated_text
        if not self._translation_looks_unfinished(raw_text, candidate):
            return candidate

        print("    ⚠️ [Translation Repair] English residue detected, retrying with strict translation...")
        for _ in range(2):
            repaired = self._strict_translate_masked_text(masked_text, mask_count, candidate)
            repaired = self._clean_hallucinations(repaired)
            if not repaired:
                continue
            if len(re.findall(r'\[MATH_MASK_\d+\]', repaired)) != mask_count:
                continue
            candidate = repaired
            if not self._translation_looks_unfinished(raw_text, candidate):
                return candidate

        sentencewise = self._sentencewise_translate_masked_text(masked_text)
        sentencewise = self._clean_hallucinations(sentencewise)
        if sentencewise and len(re.findall(r'\[MATH_MASK_\d+\]', sentencewise)) == mask_count:
            candidate = sentencewise
            if not self._translation_looks_unfinished(raw_text, candidate):
                return candidate

        formula_safe = self._translate_text_preserving_formulas(raw_text)
        if formula_safe:
            candidate = formula_safe
        return candidate

    def _strip_math_segments(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r'(?<!\\)\$(?:\\.|[^$])+(?<!\\)\$', ' ', text)

    def _formula_safety_issues(self, source_text: str, translated_text: str) -> List[str]:
        text = (translated_text or "").strip()
        if not text:
            return ["empty_translation"]

        issues = []
        source_formula_count = len(self.masker.formula_pattern.findall(source_text or ""))
        output_formula_count = len(self.masker.formula_pattern.findall(text))

        if text.count('$') % 2 != 0:
            issues.append("odd_dollar_count")
        if output_formula_count < source_formula_count:
            issues.append("formula_loss")
        if re.search(r'\[MATH_MASK_\d+\]', text):
            issues.append("placeholder_residue")

        outside_math = self._strip_math_segments(text)
        if re.search(r'\\[A-Za-z]+', outside_math):
            issues.append("latex_command_leak")
        if re.search(r'(?<!\\)[{}]', outside_math):
            issues.append("brace_leak")
        if re.search(r'(?<!\\)(?:\^|_)', outside_math):
            issues.append("script_token_leak")

        return sorted(set(issues))

    def _stabilize_formula_translation(self, raw_text: str, translated_text: str) -> Tuple[str, Dict]:
        issues = self._formula_safety_issues(raw_text, translated_text)
        if not issues:
            return translated_text, {"issues": [], "fallback_used": False, "mode": "direct"}

        print(f"    ⚠️ [Formula Guard] Unsafe translation detected: {issues}")
        safe_text = self._translate_text_preserving_formulas(raw_text)
        safe_issues = self._formula_safety_issues(raw_text, safe_text)
        if not safe_issues:
            return safe_text, {
                "issues": issues,
                "fallback_used": True,
                "mode": "formula_safe_fallback",
            }

        return translated_text, {
            "issues": issues,
            "fallback_used": False,
            "mode": "unsafe_kept",
        }

    def _correct_element_type_visually(self, item: Dict, body_base_size: float, doc: fitz.Document) -> str:
        """
        [核心纠错] 基于视觉特征修正元素类型。
        解决 1. 标题被误标为 text 2. Abstract 被误标为 author
        """
        orig_type = item.get('type', 'text').lower()
        if orig_type == 'page_footnote':
            return 'footer'
        structural_types = {'aside_text', 'page_number'}

        # 0. 绝对排除 Image/Table (防御性编程)
        if orig_type in self.IGNORE_TYPES or orig_type in structural_types:
            return orig_type

        text_content = item.get('text', '')
        text_level = item.get('text_level', None)
        
        # 1. 修正 Abstract (被误标为 Author)
        if 'author' in orig_type:
            if len(text_content) > 150 or "abstract" in text_content.lower()[:50]:
                return 'text' 
        
        # 2. 修正 Reference
        if 'reference' in orig_type and len(text_content) > 50:
            if not re.search(r'\[\d+\]', text_content) and not re.search(r'\d+\.', text_content):
                return 'text'

        # 3. [V5 修复] 先获取字号，用于更准确的类型判断
        font_size = self._get_original_metrics(item, doc)
        
        # 4. [关键] 找回丢失的 Title / Heading
        # Page 0 + 特大字号 -> 一定是 Title（优先于 text_level）
        if item['page_idx'] == 0 and font_size > body_base_size * 1.4:
            return 'title'
        
        # text_level=0 -> Title
        if text_level == 0:
            return 'title'
        
        # text_level=1 -> 根据字号区分 title 和 section_header
        if text_level == 1:
            # 字号很大（>1.4倍正文）-> title
            if font_size > body_base_size * 1.4:
                return 'title'
            # 否则是 section_header
            return 'section_header'

        # 5. 视觉字号辅助修正（没有 text_level 时）
        # 大于正文 1.15 倍 -> Heading
        if font_size > body_base_size * 1.15:
            if 'title' not in orig_type:
                return 'section_header'
            
        # 小字号 -> Caption/Footnote
        if font_size < body_base_size * 0.85:
            if 'caption' in orig_type: return 'caption'
            return 'footnote'

        return orig_type

    def _calculate_global_hierarchy_styles(self, data: List[Dict], doc: fitz.Document) -> Dict:
        """
        [关键] 计算全局字号并分配 Font Key
        """
        styles = {}
        
        # 1. Body
        body_items = [i for i in data if i.get('type') == 'text' and len(i.get('text', '')) > 20]
        if body_items:
            samples = sorted(body_items, key=lambda x: len(x['text']), reverse=True)[:15]
            src_densities = []; src_sizes = []
            for item in samples:
                w, h = item['bbox'][2]-item['bbox'][0], item['bbox'][3]-item['bbox'][1]
                src_densities.append((w*h) / len(item['text']))
                src_sizes.append(self._get_original_metrics(item, doc))
            avg_src_density = statistics.median(src_densities) if src_densities else 100
            orig_body_size = statistics.median(src_sizes) if src_sizes else 9.0
            
            page_width = float(doc[0].rect.width) if len(doc) > 0 else 0.0
            if self.layout_mode == "single_col":
                target_line, target_natural_fill = self._estimate_single_col_body_targets(body_items, page_width, avg_src_density)
            else:
                target_line, target_natural_fill = 1.35, 0.80
            natural_fill = (0.6 * (orig_body_size**2) * target_line * 1.1) / avg_src_density
            
            upscale_factor = 1.0
            if natural_fill < target_natural_fill:
                upscale_factor = min((target_natural_fill / natural_fill) ** 0.5, 1.4 if self.layout_mode == "single_col" else 1.3)
            
            golden_body_size = max(9.0, min(12.0, orig_body_size * upscale_factor))
            styles['body'] = {'size': round(golden_body_size*2)/2, 'line': target_line, 'char': 0.05, 'font_key': 'body'}
        else:
            styles['body'] = {'size': 10.5, 'line': 1.4, 'char': 0.05, 'font_key': 'body'}
            orig_body_size = 9.0
            upscale_factor = 1.0

        # 2. Others (Title, Heading, Caption...)
        # 为每种类型分配特定的 Font Key
        other_types = {
            'title': {'scale': 1.8, 'line': 1.3, 'font_key': 'title'},
            'section_header': {'scale': 1.25, 'line': 1.35, 'font_key': 'heading'},
            'page_header': {'scale': 0.9, 'line': 1.25, 'font_key': 'body'},
            'header': {'scale': 0.9, 'line': 1.25, 'font_key': 'body'},
            'footer': {'scale': 0.9, 'line': 1.25, 'font_key': 'caption'},
            'caption': {'scale': 0.9, 'line': 1.25, 'font_key': 'caption'},
            'footnote': {'scale': 0.9, 'line': 1.25, 'font_key': 'caption'},
            'figure_caption': {'scale': 0.9, 'line': 1.25, 'font_key': 'caption'},
            'table_caption': {'scale': 0.9, 'line': 1.25, 'font_key': 'caption'}
        }
        
        # [V5 新增] 预先识别可能是 title 的项（page 0 + 大字号）
        potential_titles = []
        for i in data:
            if i.get('page_idx') == 0:
                size = self._get_original_metrics(i, doc)
                if size > orig_body_size * 1.4:
                    potential_titles.append(i)
        
        for t_type, defaults in other_types.items():
            items = [i for i in data if t_type in i.get('type', '').lower()]
            
            # [V5 修复] 对于 title，如果没有显式 title 类型，使用 potential_titles
            if t_type == 'title' and not items and potential_titles:
                items = potential_titles
            
            # 基础字号计算
            if not items:
                base_sz = styles['body']['size'] * defaults['scale']
            else:
                orig_sizes = [self._get_original_metrics(i, doc) for i in items]
                base_sz = statistics.median(orig_sizes) * upscale_factor
            
            if 'title' in t_type: base_sz = min(base_sz, 26.0)
            
            styles[t_type] = {
                'size': round(base_sz * 2) / 2,
                'line': defaults['line'],
                'char': 0.05,
                'font_key': defaults['font_key'] # 注入字体键
            }

        return styles, orig_body_size

    def _calculate_target_chars(self, bbox: List[float], config: Dict) -> int:
        """
        [V4 修复] 与 _simulate_text_height 使用一致的字符宽度估算。
        
        核心思路：
        1. 使用与 _simulate_text_height 相同的字符宽度计算
        2. 确保目标字符数与高度模拟一致
        """
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= 0 or h <= 0:
            return 1
            
        size = config['size']
        line_height_ratio = config.get('line', 1.35)
        
        # 行高（像素）
        line_h = size * line_height_ratio
        
        # 计算 BBox 最多能放多少完整行
        max_lines = h / line_h
        max_lines = max(1, max_lines)
        
        # [V4 修复] 使用与 _simulate_text_height 一致的平均字符宽度
        # 假设译文 80% 中文（宽度 1.0）+ 20% 其他（宽度 0.6）
        # 平均宽度 = 0.8 * 1.0 + 0.2 * 0.6 = 0.92
        avg_char_width = size * 0.92
        
        # 每行能放的字符数
        chars_per_line = w / avg_char_width
        
        # [V4 修复] 减少换行损耗估算（从 15% 降到 5%）
        # 因为之前太保守导致目标字符数太少
        effective_chars_per_line = chars_per_line * 0.95
        
        # 目标字符数 = 每行字符数 × 行数
        target_chars = int(effective_chars_per_line * max_lines)
        
        return max(1, target_chars)
    
    def _simulate_text_height(self, text: str, bbox_width: float, config: Dict) -> float:
        """
        [V4 修复] 模拟文本排版高度，与 _calculate_target_chars 使用一致的估算。
        
        Args:
            text: 要排版的文本
            bbox_width: BBox 宽度
            config: 样式配置 (size, line)
            
        Returns:
            文本排版后的总高度（像素）
        """
        if not text or bbox_width <= 0:
            return 0
            
        size = config.get('size', 10.5)
        line_height_ratio = config.get('line', 1.35)
        line_h = size * line_height_ratio
        
        # [V4 修复] 使用与 _calculate_target_chars 一致的平均宽度
        # 简化计算：所有字符用 0.92 * size 估算
        avg_char_width = size * 0.92
        
        # 每行能放的字符数
        chars_per_line = bbox_width / avg_char_width
        
        # 计算需要多少行
        # 去除空白字符后计算有效字符数
        effective_chars = len(text.replace('\n', '').replace(' ', ''))
        
        # 需要的行数
        needed_lines = effective_chars / chars_per_line if chars_per_line > 0 else 1
        
        # 返回总高度 = 行数 × 行高
        return needed_lines * line_h

    def _tokenize_render_units(self, text: str) -> List[str]:
        tokens = []
        last = 0
        pattern = re.compile(r'(?<!\\)\$(?:\\.|[^$])+(?<!\\)\$')
        for match in pattern.finditer(text):
            if match.start() > last:
                tokens.extend(
                    re.findall(
                        r'\s+|[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*|[\u4e00-\u9fff]|.',
                        text[last:match.start()],
                    )
                )
            tokens.append(match.group(0))
            last = match.end()
        if last < len(text):
            tokens.extend(
                re.findall(
                    r'\s+|[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*|[\u4e00-\u9fff]|.',
                    text[last:],
                )
            )
        return tokens

    def _prefer_split_point(self, tokens: List[str], start: int, end: int) -> int:
        preferred = {'。', '！', '？', '；', '：', '.', '!', '?', ';', ':', '”', '"'}
        lower_bound = max(start + 1, end - 24)
        for idx in range(end, lower_bound - 1, -1):
            if tokens[idx - 1] in preferred:
                return idx
        return end

    def _estimate_split_weight(self, item: Dict) -> int:
        source_text = item.get('text', '') or ''
        formula_pattern = re.compile(r'(?<!\\)\$(?:\\.|[^$])+(?<!\\)\$')
        formula_count = len(formula_pattern.findall(source_text))
        plain_text = formula_pattern.sub(' ', source_text)
        compact = re.sub(r'\s+', '', plain_text)
        weight = len(compact) + formula_count * 6

        bbox = item.get('bbox', [0, 0, 0, 0])
        if len(bbox) == 4:
            width = max(0.0, float(bbox[2]) - float(bbox[0]))
            height = max(0.0, float(bbox[3]) - float(bbox[1]))
            if width < 120 or height < 16:
                weight = min(weight, 24)

        return max(1, weight)

    def _estimate_source_target_end(
        self,
        tokens: List[str],
        start: int,
        max_end: int,
        current_weight: int,
        remaining_weights: List[int],
    ) -> int:
        remaining_capacity = max_end - start
        total_remaining_weight = max(1, sum(remaining_weights))
        target_span = max(1, round(remaining_capacity * (current_weight / total_remaining_weight)))
        target_end = min(max_end, max(start + 1, start + target_span))
        return self._prefer_split_point(tokens, start, target_end)

    def _build_candidate_ends(
        self,
        tokens: List[str],
        start: int,
        best_end: int,
        source_target_end: int,
    ) -> List[int]:
        preferred_end = self._prefer_split_point(tokens, start, best_end)
        probe_ceiling = max(best_end, source_target_end)
        probe_floor = max(start + 1, min(best_end, source_target_end) - 48)

        candidate_ends = []
        for end in [source_target_end, preferred_end, best_end]:
            if end is not None and end >= start + 1:
                candidate_ends.append(end)
        candidate_ends.extend(range(probe_ceiling - 1, probe_floor - 1, -1))
        return candidate_ends

    def _fits_rendered_chunk(self, tool: PDFReflowTool, bbox: List[float], text: str, config: Dict) -> bool:
        candidate = (text or "").strip()
        if not candidate:
            return False
        adjusted = tool._accordion_fit_style(bbox, candidate, dict(config))
        return not tool.simulate_layout_metrics(bbox, candidate, adjusted).get('is_overflow')

    def _can_fit_suffix(self, tool: PDFReflowTool, tokens: List[str], start: int, items: List[Dict], config: Dict) -> bool:
        if not items:
            return start >= len(tokens)

        remaining_text = "".join(tokens[start:]).strip()
        if not remaining_text:
            return False

        if len(items) == 1:
            return self._fits_rendered_chunk(tool, items[0]['bbox'], remaining_text, config)

        remaining_items = len(items) - 1
        max_end = len(tokens) - remaining_items
        if max_end <= start:
            return False

        low, high = start + 1, max_end
        best_end = None
        while low <= high:
            mid = (low + high) // 2
            candidate = "".join(tokens[start:mid]).strip()
            if self._fits_rendered_chunk(tool, items[0]['bbox'], candidate, config):
                best_end = mid
                low = mid + 1
            else:
                high = mid - 1

        if best_end is None:
            return False

        preferred_end = self._prefer_split_point(tokens, start, best_end)
        probe_floor = max(start + 1, best_end - 32)
        candidate_ends = []
        for end in [preferred_end, best_end]:
            if end is not None and end >= start + 1:
                candidate_ends.append(end)
        candidate_ends.extend(range(best_end - 1, probe_floor - 1, -1))

        seen = set()
        for end in candidate_ends:
            if end in seen or end <= start:
                continue
            seen.add(end)
            candidate = "".join(tokens[start:end]).strip()
            if not self._fits_rendered_chunk(tool, items[0]['bbox'], candidate, config):
                continue
            if self._can_fit_suffix(tool, tokens, end, items[1:], config):
                return True
        return False

    def _smart_split_rendered_text(self, full_text: str, items: List[Dict], config: Dict) -> List[Tuple[int, str]]:
        if len(items) <= 1:
            return [(0, full_text.strip())]

        tool = self._get_layout_tool()
        tokens = self._tokenize_render_units(full_text)
        if not tokens:
            return [(idx, "") for idx in range(len(items))]

        split_weights = [self._estimate_split_weight(item) for item in items]
        results = []
        start = 0
        for idx, item in enumerate(items[:-1]):
            remaining_items = len(items) - idx - 1
            max_end = len(tokens) - remaining_items
            if max_end <= start:
                results.append((idx, ""))
                continue

            best_end = start + 1
            low, high = start + 1, max_end
            while low <= high:
                mid = (low + high) // 2
                candidate = "".join(tokens[start:mid]).strip()
                if self._fits_rendered_chunk(tool, item['bbox'], candidate, config):
                    best_end = mid
                    low = mid + 1
                else:
                    high = mid - 1

            source_target_end = self._estimate_source_target_end(
                tokens,
                start,
                max_end,
                split_weights[idx],
                split_weights[idx:],
            )
            candidate_ends = self._build_candidate_ends(tokens, start, best_end, source_target_end)

            chosen_end = None
            seen = set()
            for end in candidate_ends:
                if end in seen or end <= start:
                    continue
                seen.add(end)
                candidate = "".join(tokens[start:end]).strip()
                if not self._fits_rendered_chunk(tool, item['bbox'], candidate, config):
                    continue
                if self._can_fit_suffix(tool, tokens, end, items[idx + 1:], config):
                    chosen_end = end
                    break

            if chosen_end is None:
                chosen_end = source_target_end if source_target_end > start else best_end
            results.append((idx, "".join(tokens[start:chosen_end]).strip()))
            start = chosen_end

        results.append((len(items) - 1, "".join(tokens[start:]).strip()))
        return results

    def _count_chinese_chars(self, text: str) -> int:
        """统计中文字符数（去除空白和公式占位符）"""
        # 移除公式占位符
        clean_text = re.sub(r'\[MATH_MASK_\d+\]', '', text)
        clean_text = re.sub(r'\$[^$]+\$', '', clean_text)
        # 统计非空白字符
        return len(re.sub(r'\s', '', clean_text))
    
    def _translate_group(self, lid: int, group: List[Dict], global_styles: Dict, body_base_size: float, doc: fitz.Document):
        raw_text = group[0]['item'].get('context')
        if not raw_text:
            raw_text = " ".join([g['item']['text'] for g in group])
        stripped_text = raw_text.strip()
        first_item = group[0]['item']
        corrected_type = self._correct_element_type_visually(first_item, body_base_size, doc)
        is_reference_group = self._should_preserve_original_group(corrected_type, stripped_text, group)
        style_key = 'body'
        if corrected_type in global_styles:
            style_key = corrected_type
        config = global_styles.get(style_key, global_styles['body']).copy()

        group_style_infos = []
        for g in group:
            style_info = self._extract_style_info(g['item'], doc)
            group_style_infos.append(style_info)

        if group_style_infos:
            primary_style = group_style_infos[0]
            config['is_bold'] = primary_style.get('is_bold', False)
            config['is_italic'] = primary_style.get('is_italic', False)
            config['text_color'] = primary_style.get('text_color', (0, 0, 0))
            config['bg_color'] = primary_style.get('bg_color', None)
            config['original_font'] = primary_style.get('original_font', None)

        if is_reference_group:
            guard_meta = {
                "issues": [],
                "fallback_used": False,
                "mode": "preserve_original_reference",
            }
            for g in group:
                g['item']['translation_guard'] = guard_meta
            splits_with_styles = []
            for idx, g in enumerate(group):
                item_style = group_style_infos[idx] if idx < len(group_style_infos) else group_style_infos[0]
                splits_with_styles.append((idx, g['item'].get('text', ''), item_style))
            return splits_with_styles, config, 'reference'

        if not stripped_text or self._content_signal_score(stripped_text) == 0:
            guard_meta = {
                "issues": ["empty_source_text"],
                "fallback_used": True,
                "mode": "preserve_empty_source",
            }
            for g in group:
                g['item']['translation_guard'] = guard_meta
            splits_with_styles = []
            for idx, g in enumerate(group):
                item_style = group_style_infos[idx] if idx < len(group_style_infos) else (group_style_infos[0] if group_style_infos else {})
                splits_with_styles.append((idx, g['item'].get('text', ''), item_style))
            return splits_with_styles, config, corrected_type

        # [V5 新增] 检查是否有直接翻译映射（避免 API 出错）
        if stripped_text in self.DIRECT_TRANSLATION_MAP:
            direct_trans = self.DIRECT_TRANSLATION_MAP[stripped_text]
            splits_with_styles = [(0, direct_trans, group_style_infos[0] if group_style_infos else {})]
            return splits_with_styles, config, corrected_type

        masked_text, formula_map = self.masker.mask(raw_text)
        mask_count_before = len(formula_map)  # 记录占位符数量
        
        # [V7 新增] 标题类型完全跳过定长翻译
        TITLE_TYPES = {'title', 'section_header'}
        is_title_type = any(t in corrected_type for t in TITLE_TYPES)
        
        can_expand = True
        if any(t in corrected_type for t in self.NO_EXPANSION_TYPES):
            can_expand = False
        
        # [V3 核心] 标题：严格直译，禁止扩写，不加量词
        if is_title_type:
            # 标题最大长度 = 原文长度 * 0.6（中文更短）
            max_title_chars = max(3, int(len(raw_text) * 0.6))
            
            # 计算标题中的占位符数量
            title_mask_count = len(re.findall(r'\[MATH_MASK_\d+\]', masked_text))
            
            # [V9 修复] 只有当标题有占位符时才提及占位符
            if title_mask_count > 0:
                prompt = (
                    f"Translate to Chinese: {masked_text}\n\n"
                    f"RULES:\n"
                    f"1. Output ONLY the translation - no other text.\n"
                    f"2. Preserve {title_mask_count} [MATH_MASK_X] placeholders exactly.\n"
                    f"3. Keep section numbers unchanged.\n"
                    f"4. No extra words after numbers (✓'4 结果' ✗'4项结果').\n"
                    f"5. Maximum {max_title_chars} characters.\n"
                    f"6. Output format: <translation text only>\n"
                )
            else:
                prompt = (
                    f"Translate to Chinese: {masked_text}\n\n"
                    f"RULES:\n"
                    f"1. Output ONLY the translation - no other text.\n"
                    f"2. Keep section numbers unchanged.\n"
                    f"3. No extra words after numbers (✓'4 结果' ✗'4项结果').\n"
                    f"4. Maximum {max_title_chars} characters.\n"
                    f"5. Output format: <translation text only>\n"
                )
            msgs = [{"role": "user", "content": prompt}]
            trans_masked = self.client.chat_completion(msgs, temperature=0.1)
            trans_masked = self._clean_hallucinations(trans_masked) if trans_masked else ""
            
            # 去除可能的引号
            trans_masked = trans_masked.strip('"\'""''')
            
            # [V9 严格] 验证标题中的占位符，不匹配则使用原文
            title_mask_after = len(re.findall(r'\[MATH_MASK_\d+\]', trans_masked))
            if title_mask_count > 0 and title_mask_after != title_mask_count:
                print(f"    ❌ [Title Placeholder Fatal] {title_mask_count} → {title_mask_after}, using ORIGINAL")
                trans_masked = masked_text
            
            # [V3] 严格限制：如果翻译超过原文长度，强制截断
            if len(trans_masked) > len(raw_text):
                print(f"    ⚠️ Title too long ({len(trans_masked)} > {len(raw_text)}), truncating...")
                trans_masked = trans_masked[:max_title_chars]
            
            # [V3] 标题使用 rich_spans 中的原始字号
            if group_style_infos and group_style_infos[0].get('original_size'):
                config['size'] = group_style_infos[0]['original_size']
            elif first_item.get('rich_spans'):
                max_size = max((s.get('size', 10) for s in first_item.get('rich_spans', [])), default=10)
                if max_size > config.get('size', 10):
                    config['size'] = max_size
        else:
            # 非标题类型：执行定长翻译逻辑
            all_bboxes = [g['item']['bbox'] for g in group]
            
            # [V6 重构] 基于高度的视觉填充率计算
            total_bbox_height = sum([b[3] - b[1] for b in all_bboxes])
            avg_bbox_width = sum([b[2] - b[0] for b in all_bboxes]) / len(all_bboxes) if all_bboxes else 200
            
            # 目标字符数（用于提示 LLM）
            total_target_chars = sum([self._calculate_target_chars(b, config) for b in all_bboxes])
            
            # [V3] 目标填充率：95%-110%
            target_fill_min = 0.95
            target_fill_max = 1.10  # 允许 10% 溢出
            
            est_raw_len = len(raw_text) * 0.6  # 中文约为英文的 60%
            
            # [V6] 使用高度模拟预估填充率
            est_trans_text = "中" * int(est_raw_len)  # 模拟译文
            est_height = self._simulate_text_height(est_trans_text, avg_bbox_width, config)
            est_fill_ratio = est_height / total_bbox_height if total_bbox_height > 0 else 1.0
            
            # [核心修复] 带重试的翻译逻辑
            MAX_RETRIES = 3  # 增加重试次数以处理占位符问题
            trans_masked = ""
            best_result = ""
            best_fill_ratio = 0
            placeholder_valid = False  # 跟踪占位符是否完整
            
            for attempt in range(MAX_RETRIES + 1):
                if attempt == 0:
                    if not can_expand:
                        strategy = "Strategy: ACCURATE. Do NOT expand. Concise."
                    else:
                        # [V3] 95%-110% 填充目标
                        if est_fill_ratio > 1.15:
                            strategy = f"Target ~{total_target_chars} chars. Strategy: Be CONCISE but keep key information."
                        elif est_fill_ratio < 0.90:
                            # 填充不足 90%，适度扩写
                            ratio = 0.98 / est_fill_ratio if est_fill_ratio > 0 else 1.3
                            strategy = (
                                f"Target ~{total_target_chars} chars (Expand ~{ratio:.1f}x).\n"
                                "Strategy: Add some academic details to fill space."
                            )
                        else:
                            strategy = "Strategy: ACCURATE. Natural translation flow."
                else:
                    # [V3] 重试时要求达到 95% 填充
                    target_95 = int(total_target_chars * 0.98)
                    if best_fill_ratio < target_fill_min:
                        strategy = (
                            f"**LENGTH CONTROL**: Output should be approximately {target_95} Chinese characters.\n"
                            f"Current output is too short (fill ratio: {best_fill_ratio:.0%}). Target: 95%-110%.\n"
                            "Strategy: Add some details to better fill the text box."
                        )
                    else:
                        break

                # [V9 修复] 只有当原文有占位符时才提及占位符要求
                if mask_count_before > 0:
                    prompt = (
                        f"Translate to Chinese: {masked_text}\n\n"
                        f"{strategy}\n\n"
                        f"REQUIREMENTS:\n"
                        f"1. Preserve all {mask_count_before} [MATH_MASK_X] placeholders (exact format: [MATH_MASK_0]).\n"
                        f"2. Output only the translation - no labels, no explanations, no quotes.\n"
                        f"3. Keep placeholder order unchanged.\n"
                        f"4. Format: <translation text only>\n"
                    )
                else:
                    prompt = (
                        f"Translate to Chinese: {masked_text}\n\n"
                        f"{strategy}\n\n"
                        f"REQUIREMENTS:\n"
                        f"1. Output only the translation - no labels, no explanations, no quotes.\n"
                        f"2. Format: <translation text only>\n"
                    )
                
                msgs = [{"role": "user", "content": prompt}]
                trans_masked = self.client.chat_completion(msgs, temperature=0.3 + attempt * 0.05)
                if not trans_masked: 
                    continue
                
                trans_masked = self._clean_hallucinations(trans_masked)
                if not trans_masked:
                    continue
                if self._contains_spurious_meta_output(raw_text, trans_masked):
                    print(f"    ⚠️ [Meta Leak] Suspicious meta output detected (attempt {attempt+1}/{MAX_RETRIES+1})")
                    if attempt < MAX_RETRIES:
                        continue
                    trans_masked = masked_text
                
                # [关键修复] 验证占位符数量 - 如果不匹配则必须重试
                mask_count_after = len(re.findall(r'\[MATH_MASK_\d+\]', trans_masked))
                placeholder_valid = (mask_count_after == mask_count_before)
                
                if not placeholder_valid:
                    print(f"    ⚠️ [Placeholder Error] {mask_count_before - mask_count_after} placeholders lost (attempt {attempt+1}/{MAX_RETRIES+1})")
                    if attempt < MAX_RETRIES:
                        continue  # 强制重试
                    else:
                        print(f"    ❌ [Placeholder Fatal] Max retries reached, accepting incomplete result")
                
                actual_height = self._simulate_text_height(trans_masked, avg_bbox_width, config)
                actual_fill_ratio = actual_height / total_bbox_height if total_bbox_height > 0 else 1.0
                
                # 只有占位符完整时才更新最佳结果
                if placeholder_valid and actual_fill_ratio > best_fill_ratio:
                    best_result = trans_masked
                    best_fill_ratio = actual_fill_ratio
                
                # 只有占位符完整且填充率合适时才提前退出
                if placeholder_valid and target_fill_min <= actual_fill_ratio <= target_fill_max:
                    break
                
                if not can_expand or attempt >= MAX_RETRIES:
                    break
            
            trans_masked = best_result if best_result else trans_masked
            
            # [V9 严格检查] 如果占位符数量不匹配，使用原文（不翻译）
            if trans_masked:
                final_mask_count = len(re.findall(r'\[MATH_MASK_\d+\]', trans_masked))
                if final_mask_count != mask_count_before:
                    print(f"    ❌ [Placeholder Fatal] Mismatch {mask_count_before} → {final_mask_count}, using ORIGINAL text")
                    # 使用原文，但unmask公式
                    trans_masked = masked_text

            trans_masked = self._repair_untranslated_output(raw_text, masked_text, trans_masked, mask_count_before)
        
        if not trans_masked: 
            raise ValueError("Empty response after retries")

        # Unmask - 将占位符替换回公式
        final_trans = self.masker.unmask(trans_masked, formula_map)
        if self._contains_spurious_meta_output(raw_text, final_trans):
            print("    ⚠️ [Meta Leak] Falling back to original source text")
            final_trans = raw_text
        post_repair_used = False
        if self._translation_looks_unfinished(raw_text, final_trans):
            rescued = self._translate_text_preserving_formulas(raw_text)
            rescued = self._clean_hallucinations(rescued)
            if rescued and not self._contains_spurious_meta_output(raw_text, rescued) and not self._translation_looks_unfinished(raw_text, rescued):
                final_trans = rescued
                post_repair_used = True

        final_trans, guard_meta = self._stabilize_formula_translation(raw_text, final_trans)
        if post_repair_used:
            guard_meta = dict(guard_meta)
            guard_meta['fallback_used'] = True
            guard_meta['mode'] = 'unfinished_translation_repair'
            guard_meta['unfinished_repair'] = True
        for g in group:
            g['item']['translation_guard'] = guard_meta
        if len(group) == 1:
            splits = [(0, final_trans)]
        else:
            splits = self._smart_split_rendered_text(final_trans, [g['item'] for g in group], config)
        
        # [V9 最终检查] 如果还有残留占位符，强制清除
        cleaned_splits = []
        for idx, final_text in splits:
            remaining_masks = re.findall(r'\[MATH_MASK_\d+\]', final_text)
            if remaining_masks:
                print(f"    ❌ [Unmask Failed] {len(remaining_masks)} placeholders remain: {remaining_masks[:3]}")
                # 强制清除残留占位符（用空格替换）
                for mask in remaining_masks:
                    final_text = final_text.replace(mask, ' ')
                print(f"    🔧 [Auto-fix] Removed all placeholder residues")
            cleaned_splits.append((idx, final_text))
        
        splits = cleaned_splits

        # [新增] 为每个分片附加对应的样式信息
        splits_with_styles = []
        for idx, txt in splits:
            item_style = group_style_infos[idx] if idx < len(group_style_infos) else group_style_infos[0]
            splits_with_styles.append((idx, txt, item_style))

        return splits_with_styles, config, corrected_type

    def _retry_failed_blocks(self, data, groups, global_styles, body_base_size, doc):
        failed_lids = []
        for lid, group in groups.items():
            if any(not data[g['index']].get('translated') for g in group):
                first_item = group[0]['item']
                c_type = self._correct_element_type_visually(first_item, body_base_size, doc)
                if any(t in c_type for t in self.SKIP_TEXT_TYPES): continue
                if c_type in self.IGNORE_TYPES: continue
                failed_lids.append(lid)
                
        if failed_lids:
            print(f"⚠️ Retrying {len(failed_lids)} failed blocks...")
            for lid in tqdm(failed_lids, desc="Retrying"):
                try:
                    splits_with_styles, style, c_type = self._translate_group(lid, groups[lid], global_styles, body_base_size, doc)
                    for rel_idx, text_chunk, item_style in splits_with_styles:
                        real_idx = groups[lid][rel_idx]['index']
                        data[real_idx]['translated'] = text_chunk
                        data[real_idx]['is_pre_optimized'] = True
                        # 合并样式
                        merged_style = style.copy()
                        merged_style['is_bold'] = item_style.get('is_bold', False)
                        merged_style['is_italic'] = item_style.get('is_italic', False)
                        merged_style['text_color'] = item_style.get('text_color', (0, 0, 0))
                        merged_style['bg_color'] = item_style.get('bg_color', None)
                        data[real_idx]['golden_style'] = merged_style
                        data[real_idx]['detected_type'] = c_type
                    if self.planner:
                        self.planner.mark_translation_ready(
                            lid,
                            metadata={
                                'detected_type': c_type,
                                'bucket': c_type,
                                'translation_status': 'retry_ready',
                                'translation_guard': groups[lid][0]['item'].get('translation_guard', {}),
                            },
                        )
                except Exception as e:
                    print(f"    ❌ [Retry Failed] lid={lid}: {e}")

    def process_documents(self, data: List[Dict], pdf_path: str):
        print(f"🚀 [TranslationAgent] Processing {pdf_path}...")
        doc = fitz.open(pdf_path)
        recovery_summary = recover_low_information_items(data, doc)
        if recovery_summary.get('repaired'):
            print(
                f"🩹 [TranslationAgent] Recovered {recovery_summary['repaired']} low-information text blocks "
                f"from source text layers."
            )
        page_width = doc[0].rect.width
        
        body_items = [i for i in data if i.get('type') == 'text' and len(i.get('text','')) > 20]
        self.layout_mode = self._detect_layout_mode(body_items, page_width)
        print(f"🎯 [Layout] Mode: {self.layout_mode}")
        
        global_styles, body_base_size = self._calculate_global_hierarchy_styles(data, doc)
        self.global_style_config = global_styles

        if self.planner:
            self.planner.build_initial_plan(
                data,
                pdf_path,
                layout_mode=self.layout_mode,
                global_styles=global_styles,
                body_base_size=body_base_size,
            )
        
        groups = {}
        for i, item in enumerate(data):
            # [关键] 绝对跳过 Image / Table
            if item.get('type') in self.IGNORE_TYPES:
                continue
                
            lid = item.get('logical_para_id')
            if lid is None: continue
            if lid not in groups: groups[lid] = []
            groups[lid].append({'index': i, 'item': item})
            
        sorted_lids = sorted(groups.keys())
        self._annotate_reference_sections(groups, sorted_lids, body_base_size, doc, data=data)
        pbar = tqdm(sorted_lids, desc="Translating", unit="block")
        for lid in pbar:
            group = groups[lid]; group.sort(key=lambda x: (x['item']['page_idx'], x['item']['bbox'][1]))
            if data[group[0]['index']].get('translated'): continue
            first_item = group[0]['item']; raw_type = first_item.get('type','').lower()
            if any(t in raw_type for t in self.SKIP_TEXT_TYPES):
                 if not ('author' in raw_type and len(first_item.get('text','')) > 100): continue
            try:
                splits_with_styles, style, c_type = self._translate_group(lid, group, global_styles, body_base_size, doc)
                for rel_idx, text_chunk, item_style in splits_with_styles:
                    real_idx = group[rel_idx]['index']
                    data[real_idx]['translated'] = text_chunk
                    data[real_idx]['is_pre_optimized'] = True
                    # 合并全局样式与该块的原始样式
                    merged_style = style.copy()
                    merged_style['is_bold'] = item_style.get('is_bold', False)
                    merged_style['is_italic'] = item_style.get('is_italic', False)
                    merged_style['text_color'] = item_style.get('text_color', (0, 0, 0))
                    merged_style['bg_color'] = item_style.get('bg_color', None)
                    data[real_idx]['golden_style'] = merged_style
                    data[real_idx]['detected_type'] = c_type
                if self.planner:
                    self.planner.mark_translation_ready(
                        lid,
                        metadata={
                            'detected_type': c_type,
                            'bucket': c_type,
                            'translation_status': 'ready',
                            'translation_guard': group[0]['item'].get('translation_guard', {}),
                        },
                    )
            except Exception as e:
                print(f"    ❌ [Translate Failed] lid={lid}: {e}")
        
        self._retry_failed_blocks(data, groups, global_styles, body_base_size, doc)
        if self.planner:
            self.planner.finalize_m0(data)
        doc.close()
        return data
