# tools/text_refiner.py
# [V8 重构] 保留富文本样式信息

import math
import re
import json
from typing import List, Dict, Any, Tuple
from tools.api_client import APIClient
from tools.pdf_reflow_tool import sanitize_inline_math_text

class TextRefiner:
    def __init__(self, api_client: APIClient):
        self.client = api_client
        # 定义成对标点，切分时绝不能打断
        self.PAIRS = {
            '(': ')', '（': '）',
            '[': ']', '【': '】',
            '{': '}', 
            '"': '"', '"': '"',
            '《': '》'
        }
        self.SENTENCE_END = {'.', '。', '!', '！', '?', '？'}
        self.CLAUSE_END = {',', '，', ';', '；', ':', '：'}
        
        # [V8 新增] 公式遮罩
        self._formula_map = {}
        self._mask_counter = 0

    def _extract_math_anchors(self, text: str) -> List[str]:
        return re.findall(r'(?<!\\)\$(?:\\.|[^$])+(?<!\\)\$', text)
    
    # ========== [V8 新增] 公式遮罩和样式对齐方法 ==========
    
    def _mask_formulas(self, text: str) -> Tuple[str, Dict[str, str]]:
        """遮罩 LaTeX 公式，返回遮罩后文本和映射表"""
        formula_map = {}
        
        def replace_formula(match):
            formula = match.group(0)
            mask = f"[MATH_{len(formula_map)}]"
            formula_map[mask] = formula
            return mask
        
        # 匹配 $...$ 和 $$...$$ 公式
        masked = re.sub(r'\$\$[^$]+\$\$|\$[^$]+\$', replace_formula, text)
        return masked, formula_map
    
    def _unmask_formulas(self, text: str, formula_map: Dict[str, str]) -> str:
        """还原被遮罩的公式"""
        result = text
        for mask, formula in formula_map.items():
            result = result.replace(mask, formula)
        return sanitize_inline_math_text(result)

    def _strip_math_segments(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r'(?<!\\)\$(?:\\.|[^$])+(?<!\\)\$', ' ', text)

    def _rewrite_has_formula_safety_issues(self, source_text: str, rewritten_text: str, formula_map: Dict[str, str]) -> bool:
        text = (rewritten_text or "").strip()
        if not text:
            return True

        expected_masks = len(formula_map)
        actual_masks = len(re.findall(r'\[MATH_\d+\]', text))
        if actual_masks != expected_masks:
            return True

        unmasked = self._unmask_formulas(text, formula_map)
        source_formula_count = len(re.findall(r'(?<!\\)\$(?:\\.|[^$])+(?<!\\)\$', source_text or ""))
        output_formula_count = len(re.findall(r'(?<!\\)\$(?:\\.|[^$])+(?<!\\)\$', unmasked))
        if unmasked.count('$') % 2 != 0 or output_formula_count < source_formula_count:
            return True

        outside_math = self._strip_math_segments(unmasked)
        if re.search(r'\\[A-Za-z]+', outside_math):
            return True
        if re.search(r'(?<!\\)[{}]', outside_math):
            return True
        if re.search(r'(?<!\\)(?:\^|_)', outside_math):
            return True

        return False
    
    def _extract_style_keywords(self, rich_spans: List[Dict]) -> List[Tuple[str, bool, bool]]:
        """从 rich_spans 提取有特殊样式的词（粗体/斜体）"""
        keywords = []
        for span in rich_spans:
            text = span.get('text', '').strip()
            if not text or len(text) < 2:
                continue
            flags = span.get('flags', 0)
            is_bold = bool(flags & 16)
            is_italic = bool(flags & 2)
            if is_bold or is_italic:
                keywords.append((text, is_bold, is_italic))
        return keywords
    
    def _semantic_align_styles(self, source_text: str, target_text: str, 
                               style_keywords: List[Tuple[str, bool, bool]]) -> Dict[str, Tuple[bool, bool]]:
        """
        [V8 核心] 使用 LLM 进行语义样式对齐
        
        输入: 原文、译文、原文中有样式的词列表
        输出: {译文中对应词: (is_bold, is_italic)}
        """
        if not style_keywords:
            return {}
        
        # 构建样式词描述
        style_desc = []
        for word, is_bold, is_italic in style_keywords:
            styles = []
            if is_bold: styles.append("bold")
            if is_italic: styles.append("italic")
            style_desc.append(f"'{word}' ({','.join(styles)})")
        
        prompt = f"""Given original English text and its Chinese translation, find the Chinese translations of styled words.

Original: {source_text[:500]}
Translation: {target_text[:500]}

Styled words in original: {', '.join(style_desc)}

For each styled word, output the corresponding Chinese word/phrase in the translation.
Format: original_word -> chinese_translation
Only output the mappings, one per line. If a word has no clear translation, skip it."""

        try:
            response = self.client.chat_completion([{"role": "user", "content": prompt}], temperature=0.1)
            if not response:
                return {}
            
            # 解析响应
            style_mapping = {}
            for line in response.strip().split('\n'):
                if '->' not in line:
                    continue
                parts = line.split('->')
                if len(parts) != 2:
                    continue
                orig_word = parts[0].strip().strip("'\"")
                trans_word = parts[1].strip().strip("'\"")
                
                # 查找原词的样式
                for word, is_bold, is_italic in style_keywords:
                    if word.lower() == orig_word.lower() or orig_word.lower() in word.lower():
                        if trans_word and trans_word in target_text:
                            style_mapping[trans_word] = (is_bold, is_italic)
                        break
            
            return style_mapping
        except Exception:
            return {}

    def _is_pair_start(self, char): return char in self.PAIRS
    def _is_pair_end(self, char, stack): 
        if not stack: return False
        return char == self.PAIRS.get(stack[-1])

    def _get_split_score(self, char, distance_to_target):
        """
        计算切分点的得分：越低越好
        """
        base_score = abs(distance_to_target) # 基础分是距离
        
        # 优先级奖励 (减分)
        if char in self.SENTENCE_END:
            base_score -= 15 # 最优先在句号切分
        elif char in self.CLAUSE_END:
            base_score -= 5  # 其次在逗号切分
        
        return base_score

    def _smart_distribute_semantic(self, full_trans: str, siblings: List[Dict]) -> List[str]:
        """
        [专家算法 v3.0] 语义感知 + 锚点保护 + 比例分发
        """
        if not full_trans: return [""] * len(siblings)
        
        # 1. 计算目标比例
        orig_lens = [len(x.get('text', '')) for x in siblings]
        total_len = sum(orig_lens)
        ratios = [l / total_len for l in orig_lens] if total_len > 0 else [1/len(siblings)]*len(siblings)
        
        # 2. 提取锚点 (公式)
        block_anchors = []
        for sib in siblings:
            formulas = self._extract_math_anchors(sib.get('text', ''))
            block_anchors.append(formulas)

        # 3. Tokenize (保持公式原子性)
        tokens = []
        # 分割正则：公式 OR 标点 OR 普通字符
        # 这里为了精细控制，我们将文本打散，但保持公式完整
        pattern = r'((?<!\\)\$(?:\\.|[^$])+(?<!\\)\$|[^$])' 
        # 注意：上面的正则会将每个汉字拆开，方便寻找标点
        
        # 为了效率，我们先拆出公式块，再拆出字符
        split_chunks = re.split(r'((?<!\\)\$(?:\\.|[^$])+(?<!\\)\$)', full_trans)
        for chunk in split_chunks:
            if not chunk: continue
            if chunk.startswith('$') and chunk.endswith('$'):
                tokens.append(chunk)
            else:
                tokens.extend(list(chunk))

        total_tokens = len(tokens)
        distributed_texts = []
        current_idx = 0
        
        # 4. 智能分发循环
        for i in range(len(siblings)):
            # 最后一个块直接拿走剩余
            if i == len(siblings) - 1:
                remaining = "".join(tokens[current_idx:])
                distributed_texts.append(remaining)
                break
            
            target_count = int(total_tokens * ratios[i])
            anchors_needed = block_anchors[i]
            
            # 搜索窗口：在目标长度的 [0.8, 1.3] 范围内寻找最佳切分点
            min_len = int(target_count * 0.8)
            # 上限稍微放宽，但不能超过剩余总量的太多
            max_len = int(target_count * 1.3)
            
            best_split_idx = -1
            best_score = float('inf')
            
            pair_stack = []
            anchors_found = 0
            current_anchors_in_window = 0
            
            # 模拟向前扫描
            scan_cursor = current_idx
            count = 0
            
            while scan_cursor < total_tokens:
                token = tokens[scan_cursor]
                count += 1
                
                # A. 维护成对符号栈
                if self._is_pair_start(token):
                    pair_stack.append(token)
                elif self._is_pair_end(token, pair_stack):
                    pair_stack.pop()
                
                # B. 维护锚点状态
                if token in anchors_needed:
                    anchors_found += 1
                
                # C. 判断是否在候选窗口内
                if count >= min_len:
                    # 必须满足硬性条件：
                    # 1. 栈为空 (不在括号/引号内)
                    # 2. 必须尽量包含所需锚点 (除非已经找不到了)
                    
                    # 简单的锚点检查：如果还缺锚点，尽量往后延
                    missing_anchors = (anchors_found < len(anchors_needed))
                    
                    if not pair_stack:
                        # 计算得分
                        score = self._get_split_score(token, count - target_count)
                        
                        # 如果缺锚点，大幅增加分数(惩罚)，迫使往后找
                        if missing_anchors: 
                            score += 50
                        
                        if score < best_score:
                            best_score = score
                            best_split_idx = scan_cursor + 1 # +1 包含当前标点
                
                # D. 超过最大长度，强制停止搜索
                if count >= max_len:
                    break
                    
                scan_cursor += 1
            
            # E. 确定切分点
            if best_split_idx == -1:
                # 没找到好点 (可能整个段落都在括号里，或者没标点)
                # 强制回退到 target_count (硬切分)
                # 此时也要注意不要切断公式 (tokens已经是原子的了，所以不会切断公式内部)
                split_len = min(target_count, total_tokens - current_idx)
                best_split_idx = current_idx + split_len
            
            # 执行切分
            chunk = tokens[current_idx : best_split_idx]
            distributed_texts.append("".join(chunk))
            current_idx = best_split_idx

        # 5. 兜底空检查
        for i in range(len(distributed_texts)):
            if not distributed_texts[i] and orig_lens[i] > 0:
                distributed_texts[i] = " " 
                
        return distributed_texts

    def rewrite_logical_block(self, idx: int, data: List[Dict], mode: str) -> List[Tuple[int, str, Dict]]:
        """
        [V8 重构] 重写逻辑块，保留富文本样式信息
        
        返回: [(item_idx, new_text, style_mapping), ...]
              style_mapping: {译文词: (is_bold, is_italic)}
        """
        item = data[idx]
        lid = item.get('logical_para_id')
        if lid is None: return []

        siblings = [x for x in data if x.get('logical_para_id') == lid and x.get('type') == item.get('type')]
        siblings.sort(key=lambda x: (x['page_idx'], x['bbox'][1], x['bbox'][0]))
        sibling_indices = [data.index(sib) for sib in siblings]

        full_orig = siblings[0].get('context', " ".join([x.get('text', '') for x in siblings]))
        full_trans = "".join([x.get('translated', '') for x in siblings])
        if not full_trans: return []
        
        # [V8 新增] 收集所有 siblings 的 rich_spans
        all_rich_spans = []
        for sib in siblings:
            spans = sib.get('rich_spans', [])
            all_rich_spans.extend(spans)
        
        # [V8 新增] 提取原文中的样式关键词
        style_keywords = self._extract_style_keywords(all_rich_spans)
        
        # [V8 新增] 遮罩公式
        masked_trans, formula_map = self._mask_formulas(full_trans)

        return self.rewrite_logical_block_with_guidance(idx, data, mode=mode)

    def rewrite_logical_block_with_guidance(
        self,
        idx: int,
        data: List[Dict],
        mode: str,
        target_chars: int = 0,
        current_chars: int = 0,
        target_fill_ratio: float = 0.0,
        current_fill_ratio: float = 0.0,
    ) -> List[Tuple[int, str, Dict]]:
        item = data[idx]
        lid = item.get('logical_para_id')
        if lid is None:
            return []

        siblings = [x for x in data if x.get('logical_para_id') == lid and x.get('type') == item.get('type')]
        siblings.sort(key=lambda x: (x['page_idx'], x['bbox'][1], x['bbox'][0]))
        sibling_indices = [data.index(sib) for sib in siblings]

        full_orig = siblings[0].get('context', " ".join([x.get('text', '') for x in siblings]))
        full_trans = "".join([x.get('translated', '') for x in siblings])
        if not full_trans:
            return []
        
        all_rich_spans = []
        for sib in siblings:
            spans = sib.get('rich_spans', [])
            all_rich_spans.extend(spans)
        style_keywords = self._extract_style_keywords(all_rich_spans)
        masked_trans, formula_map = self._mask_formulas(full_trans)

        effective_current_chars = current_chars or len(re.sub(r'\s', '', full_trans))
        guidance = ""
        if target_chars > 0:
            guidance = (
                f"Current length is about {effective_current_chars} characters. "
                f"Rewrite toward about {target_chars} Chinese characters. "
            )

        if mode == 'shorten':
            target_action = (
                guidance +
                "Condense the translation to better fit the layout while preserving core meaning and academic tone."
            )
        else:
            target_action = (
                guidance +
                "Expand the translation to better fill the visual space using formal academic phrasing. "
                "Do NOT add new facts, only elaborate on existing meaning."
            )
        if target_fill_ratio > 0 and current_fill_ratio > 0:
            target_action += f" Current fill ratio is {current_fill_ratio:.2f}; target fill ratio is about {target_fill_ratio:.2f}."

        prompt = (
            f"Original: {full_orig}\n"
            f"Current Translation: {masked_trans}\n"
            f"Goal: {target_action}\n"
            "Rules:\n"
            "1. KEEP all [MATH_n] placeholders exactly as is.\n"
            "2. Maintain academic tone.\n"
            "3. Return ONLY the revised text.\n"
        )

        messages = [{"role": "user", "content": prompt}]
        new_text = self.client.chat_completion(messages)
        if not new_text: return []
        new_text = new_text.replace("```", "").strip()

        if self._rewrite_has_formula_safety_issues(full_trans, new_text, formula_map):
            return []
        
        # [V8 新增] 还原公式
        new_text = self._unmask_formulas(new_text, formula_map)
        
        # [V8 新增] 进行语义样式对齐
        style_mapping = {}
        if style_keywords:
            style_mapping = self._semantic_align_styles(full_orig, new_text, style_keywords)

        # 使用语义分发
        try:
            distributed_texts = self._smart_distribute_semantic(new_text, siblings)
        except Exception:
            # 简单回退逻辑...
            total = len(new_text)
            ratios = [len(x.get('text',''))/len(full_orig) if len(full_orig)>0 else 1/len(siblings) for x in siblings]
            distributed_texts = []
            curr = 0
            for r in ratios:
                l = int(total*r)
                distributed_texts.append(new_text[curr:curr+l])
                curr+=l
            if len(distributed_texts)<len(ratios): distributed_texts.append(new_text[curr:])

        # [V8 修改] 返回包含样式映射的结果
        updates = []
        for i, real_idx in enumerate(sibling_indices):
            text = distributed_texts[i] if i < len(distributed_texts) else " "
            # 为每个分段提取相关的样式映射
            segment_style = {k: v for k, v in style_mapping.items() if k in text}
            updates.append((real_idx, text, segment_style))
            
        return updates

    def rewrite_column_group(
        self,
        data: List[Dict],
        group_specs: List[Dict[str, Any]],
        target_fill_ratio: float = 0.95,
    ) -> Dict[int, List[Tuple[int, str, Dict]]]:
        """
        Jointly rewrite multiple logical blocks in the same column.

        group_specs:
            [
                {
                    "lid": 12,
                    "mode": "shorten",
                    "target_chars": 120,
                    "issue": "OVERFLOW",
                },
                ...
            ]
        """
        if not group_specs:
            return {}

        lid_to_spec = {spec["lid"]: spec for spec in group_specs}
        lid_to_siblings = {}
        block_payloads = []

        for spec in group_specs:
            lid = spec["lid"]
            siblings = [x for x in data if x.get('logical_para_id') == lid]
            siblings.sort(key=lambda x: (x.get('page_idx', 0), x.get('bbox', [0, 0, 0, 0])[1], x.get('bbox', [0, 0, 0, 0])[0]))
            if not siblings:
                continue
            lid_to_siblings[lid] = siblings
            full_orig = siblings[0].get('context', " ".join([x.get('text', '') for x in siblings]))
            full_trans = "".join([x.get('translated', '') for x in siblings]).strip()
            if not full_trans:
                continue
            masked_trans, _ = self._mask_formulas(full_trans)
            block_payloads.append(
                {
                    "lid": lid,
                    "mode": spec.get("mode", "shorten"),
                    "issue": spec.get("issue", ""),
                    "target_chars": spec.get("target_chars", len(re.sub(r'\s', '', full_trans))),
                    "original": full_orig,
                    "translation": masked_trans,
                }
            )

        if len(block_payloads) < 2:
            return {}

        prompt_lines = [
            "You are jointly rewriting multiple Chinese translations from the same PDF column.",
            "Optimize them together so the whole column is more balanced.",
            "Rules:",
            "1. Keep all [MATH_n] placeholders unchanged.",
            "2. Keep facts unchanged.",
            "3. Return ONLY valid JSON.",
            "4. JSON format: {\"blocks\": [{\"lid\": 1, \"text\": \"...\"}, ...]}",
            "",
        ]

        for payload in block_payloads:
            action_desc = "shorten" if payload["mode"] == "shorten" else "lengthen"
            prompt_lines.extend(
                [
                    f"LID: {payload['lid']}",
                    f"Issue: {payload['issue']}",
                    f"Desired action: {action_desc}",
                    f"Target chars: {payload['target_chars']}",
                    f"Original: {payload['original']}",
                    f"Current Translation: {payload['translation']}",
                    "",
                ]
            )
        prompt_lines.append(f"Column target fill ratio: {target_fill_ratio:.2f}")
        prompt = "\n".join(prompt_lines)

        response = self.client.chat_completion([{"role": "user", "content": prompt}], temperature=0.2)
        if not response:
            return {}

        cleaned = response.replace("```json", "").replace("```", "").strip()
        try:
            payload = json.loads(cleaned)
        except Exception:
            return {}

        updates_by_lid: Dict[int, List[Tuple[int, str, Dict]]] = {}
        for block in payload.get("blocks", []):
            lid = block.get("lid")
            if lid not in lid_to_siblings:
                continue
            new_text = str(block.get("text", "")).strip()
            if not new_text:
                continue

            siblings = lid_to_siblings[lid]
            sibling_indices = [data.index(sib) for sib in siblings]
            all_rich_spans = []
            for sib in siblings:
                all_rich_spans.extend(sib.get('rich_spans', []))
            style_keywords = self._extract_style_keywords(all_rich_spans)
            _, formula_map = self._mask_formulas("".join([sib.get('translated', '') for sib in siblings]))
            if self._rewrite_has_formula_safety_issues("".join([sib.get('translated', '') for sib in siblings]), new_text, formula_map):
                continue
            unmasked_text = self._unmask_formulas(new_text, formula_map)
            style_mapping = self._semantic_align_styles(
                siblings[0].get('context', " ".join([x.get('text', '') for x in siblings])),
                unmasked_text,
                style_keywords,
            ) if style_keywords else {}

            try:
                distributed_texts = self._smart_distribute_semantic(unmasked_text, siblings)
            except Exception:
                distributed_texts = [unmasked_text] + [""] * (len(siblings) - 1)

            updates = []
            for i, real_idx in enumerate(sibling_indices):
                text = distributed_texts[i] if i < len(distributed_texts) else " "
                segment_style = {k: v for k, v in style_mapping.items() if k in text}
                updates.append((real_idx, text, segment_style))
            updates_by_lid[lid] = updates

        return updates_by_lid
