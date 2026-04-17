import statistics
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Dict, List, Optional

import fitz


class PlannerAgent:
    """
    Central planner for document profiling, M0 golden-style freezing,
    strict block-state transitions, and fine-grained attribution logging.
    """

    VALID_STATES = {
        "REGISTERED",
        "TRANSLATED",
        "PENDING",
        "OVERFLOW",
        "UNDERFLOW",
        "REWRITE_SCHEDULED",
        "OPTIMAL",
        "FORCE_FIT",
    }

    STABLE_STATES = {"OPTIMAL", "FORCE_FIT"}

    ALLOWED_TRANSITIONS = {
        "REGISTERED": {"REGISTERED", "TRANSLATED"},
        "TRANSLATED": {"TRANSLATED", "PENDING"},
        "PENDING": {"PENDING", "OVERFLOW", "UNDERFLOW", "OPTIMAL", "FORCE_FIT", "REWRITE_SCHEDULED"},
        "OVERFLOW": {"OVERFLOW", "REWRITE_SCHEDULED", "OPTIMAL", "FORCE_FIT", "PENDING"},
        "UNDERFLOW": {"UNDERFLOW", "REWRITE_SCHEDULED", "OPTIMAL", "FORCE_FIT", "PENDING"},
        "REWRITE_SCHEDULED": {"REWRITE_SCHEDULED", "PENDING", "OPTIMAL", "FORCE_FIT"},
        "OPTIMAL": {"OPTIMAL"},
        "FORCE_FIT": {"FORCE_FIT"},
    }

    BUCKET_ALIASES = {
        "text": "body",
        "abstract": "body",
        "body": "body",
        "title": "title",
        "section_header": "section_header",
        "caption": "caption",
        "figure_caption": "caption",
        "table_caption": "caption",
        "footnote": "footnote",
        "page_header": "header",
        "header": "header",
        "page_footer": "footer",
        "footer": "footer",
    }

    REWRITE_PRESERVE_TYPES = {
        "reference",
        "equation",
        "code",
        "url",
        "page_header",
        "page_footer",
        "aside_text",
        "page_number",
        "author",
        "affiliation",
    }

    DEFAULT_STYLE_POLICY = {
        "lock_font_size": True,
        "lock_line_spacing": True,
        "char_spacing_abs_min": 0.0,
        "char_spacing_abs_max": 0.2,
        "char_spacing_max_global_spread": 0.06,
        "char_spacing_max_delta_from_bucket": 0.035,
        "line_spacing_abs_min": 1.15,
        "line_spacing_abs_max": 1.55,
        "line_spacing_max_delta_from_bucket": 0.05,
        "body_probe_safe_fill": 0.974,
        "dense_body_probe_safe_fill": 0.952,
        "other_probe_safe_fill": 0.978,
        "dense_probe_safe_box_ratio": 0.90,
        "body_probe_safe_box_ratio": 0.962,
        "other_probe_safe_box_ratio": 0.986,
        "body_freeze_quantile": 0.25,
        "body_freeze_quantile_spread1": 0.2,
        "body_freeze_quantile_spread2": 0.15,
        "rewrite_conservative_doc_math_ratio": 0.32,
        "rewrite_disable_doc_math_ratio": 0.48,
        "rewrite_disable_doc_multibox_ratio": 0.22,
        "rewrite_disable_safe_body_ratio": 0.08,
        "rewrite_disable_safe_body_min_blocks": 8,
        "rewrite_block_max_chars": 360,
        "safe_prose_rewrite_min_chars": 20,
    }

    def __init__(self, target_lang: str = "zh", style_policy: Optional[Dict] = None):
        self.target_lang = target_lang
        self.style_policy = deepcopy(self.DEFAULT_STYLE_POLICY)
        if style_policy:
            self.style_policy.update(style_policy)
        self.document_profile: Dict = {}
        self.global_plan: Dict = {}
        self.block_states: Dict[str, Dict] = {}
        self.action_log: List[Dict] = []
        self.round_history: List[Dict] = []
        self.action_counter: Counter = Counter()
        self.issue_counter: Counter = Counter()
        self.transition_log: List[Dict] = []
        self.block_attribution: Dict[str, Dict] = {}
        self.m0_probe_log: List[Dict] = []
        self._current_round: Optional[Dict] = None
        self._layout_tool = None

    def build_initial_plan(
        self,
        data: List[Dict],
        pdf_path: str,
        layout_mode: Optional[str] = None,
        global_styles: Optional[Dict] = None,
        body_base_size: Optional[float] = None,
    ) -> Dict:
        self.document_profile = self._analyze_document(data, pdf_path)
        self.ensure_block_registry(data)

        if layout_mode:
            self.document_profile["layout_mode"] = layout_mode

        initial_styles = deepcopy(global_styles) if global_styles else {}
        self.global_plan = {
            "target_language": self.target_lang,
            "layout_mode": layout_mode or self.document_profile.get("layout_mode", "single_col"),
            "body_base_size": body_base_size,
            "initial_golden_styles": deepcopy(initial_styles),
            "golden_styles": deepcopy(initial_styles),
            "golden_font_size": self._extract_golden_font_size(initial_styles),
            "lock_font_size": bool(self.style_policy.get("lock_font_size", True)),
            "m0_locked": False,
            "m0_status": "profiled",
        }
        self._annotate_document_risk(data)
        self._annotate_block_risks(data)
        return self.export_summary()

    def ensure_block_registry(self, data: List[Dict]) -> None:
        for item in data:
            lid = item.get("logical_para_id")
            if lid is None:
                continue

            key = str(lid)
            block_info = self.block_states.setdefault(
                key,
                {
                    "state": "REGISTERED",
                    "round": 0,
                    "reason": "registered",
                    "history": [],
                    "meta": {},
                    "m0": {},
                },
            )
            block_info["meta"].update(
                {
                    "type": item.get("type", ""),
                    "page_idx": item.get("page_idx", item.get("page", 0)),
                }
            )
            self._ensure_block_attribution(key)

    def mark_translation_ready(self, lid, metadata: Optional[Dict] = None) -> None:
        self.update_block_state(
            lid,
            "TRANSLATED",
            round_index=0,
            reason="translation_ready",
            metadata=metadata,
        )

    def finalize_m0(self, data: List[Dict]) -> Dict:
        """
        Real M0 stage:
        1. Probe translated blocks to estimate local best-fit sizes.
        2. Aggregate robust frozen sizes by semantic bucket.
        3. Freeze golden styles for all later rounds.
        """
        self.ensure_block_registry(data)

        bucket_probes: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"size": [], "line": [], "char": []}
        )
        block_probe_values: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"size": [], "line": [], "char": []}
        )
        self.m0_probe_log = []

        for idx, item in enumerate(data):
            translated = item.get("translated", "").strip()
            bbox = item.get("bbox", [])
            golden_style = item.get("golden_style")
            lid = item.get("logical_para_id")
            if lid is None or not translated or len(bbox) != 4 or not golden_style:
                continue

            bucket = self._resolve_bucket(item, golden_style)
            probe = self._probe_block_style(item, bucket, golden_style)
            probe_record = {
                "index": idx,
                "lid": lid,
                "bucket": bucket,
                "probe_size": probe["size"],
                "fill_ratio": probe["fill_ratio"],
                "overflow_px": probe["overflow_px"],
                "score": probe["score"],
            }
            self.m0_probe_log.append(probe_record)
            item["m0_bucket"] = bucket
            item["m0_probe"] = deepcopy(probe_record)

            if probe["size"] is not None:
                bucket_probes[bucket]["size"].append(probe["size"])
                block_probe_values[str(lid)]["size"].append(probe["size"])
            if probe.get("line") is not None:
                bucket_probes[bucket]["line"].append(probe["line"])
                block_probe_values[str(lid)]["line"].append(probe["line"])
        frozen_bucket_styles = self._freeze_bucket_styles(bucket_probes)
        initial_styles = deepcopy(
            self.global_plan.get("initial_golden_styles")
            or self.global_plan.get("golden_styles")
            or {}
        )
        frozen_styles = self._apply_frozen_bucket_styles(initial_styles, frozen_bucket_styles)
        char_guardrails = self._build_char_spacing_guardrails(frozen_bucket_styles, fallback_styles=frozen_styles)
        frozen_styles = self._clamp_frozen_style_chars(frozen_styles, char_guardrails)

        for item in data:
            golden_style = item.get("golden_style")
            if not golden_style:
                continue
            bucket = item.get("m0_bucket") or self._resolve_bucket(item, golden_style)
            bucket_style = frozen_bucket_styles.get(bucket, {})
            frozen_style = deepcopy(golden_style)
            if bucket_style.get("size") is not None:
                frozen_style["size"] = bucket_style["size"]
            if bucket_style.get("line") is not None:
                frozen_style["line"] = bucket_style["line"]
            if frozen_style.get("char") is not None:
                frozen_style["char"] = self._clamp_char_with_guardrails(frozen_style["char"], char_guardrails)
            item["pre_m0_golden_style"] = deepcopy(golden_style)
            item["golden_style"] = frozen_style
            item["m0_frozen_style"] = deepcopy(frozen_style)

        for lid, block_info in self.block_states.items():
            probe_values = block_probe_values.get(lid, {})
            probe_sizes = probe_values.get("size", [])
            probe_lines = probe_values.get("line", [])
            probe_chars = probe_values.get("char", [])
            block_info["m0"] = {
                "probe_count": len(probe_sizes),
                "probe_sizes": deepcopy(probe_sizes),
                "probe_lines": deepcopy(probe_lines),
                "probe_chars": deepcopy(probe_chars),
                "frozen_bucket": block_info["meta"].get("bucket"),
                "frozen_size": self._round_half(
                    statistics.median(probe_sizes)
                ) if probe_sizes else None,
                "frozen_line": self._round_to_step(
                    statistics.median(probe_lines),
                    0.05,
                ) if probe_lines else None,
                "frozen_char": self._round_to_step(
                    statistics.median(probe_chars),
                    0.005,
                ) if probe_chars else None,
            }
            if block_info["state"] == "TRANSLATED":
                self.update_block_state(
                    lid,
                    "PENDING",
                    round_index=0,
                    reason="m0_locked",
                    metadata={
                        "m0_probe_count": len(probe_sizes),
                        "m0_bucket": block_info["meta"].get("bucket"),
                    },
                )

        self.global_plan["golden_styles"] = deepcopy(frozen_styles)
        self.global_plan["frozen_golden_styles"] = deepcopy(frozen_styles)
        self.global_plan["golden_font_size"] = self._extract_golden_font_size(frozen_styles)
        self.global_plan["golden_line_spacing"] = self._extract_golden_line_spacing(frozen_styles)
        self.global_plan["char_spacing_guardrails"] = deepcopy(char_guardrails)
        self.global_plan["m0_locked"] = True
        self.global_plan["m0_status"] = "locked"
        self.global_plan["m0_probe_summary"] = {
            "probe_count": len(self.m0_probe_log),
            "bucket_probe_count": {
                bucket: {
                    "size": len(values.get("size", [])),
                    "line": len(values.get("line", [])),
                    "char": len(values.get("char", [])),
                }
                for bucket, values in bucket_probes.items()
            },
            "bucket_frozen_styles": deepcopy(frozen_bucket_styles),
        }
        return self.export_summary()

    def start_round(self, round_index: int) -> None:
        self._current_round = {
            "round": round_index,
            "processed_lids": set(),
            "skipped_optimal": 0,
            "actions": Counter(),
            "issues": Counter(),
            "results": Counter(),
            "events": 0,
            "transitions": Counter(),
        }

    def finish_round(self, round_index: int) -> Dict:
        if not self._current_round or self._current_round["round"] != round_index:
            return {}

        summary = {
            "round": round_index,
            "processed_blocks": len(self._current_round["processed_lids"]),
            "skipped_optimal": self._current_round["skipped_optimal"],
            "actions": dict(self._current_round["actions"]),
            "issues": dict(self._current_round["issues"]),
            "results": dict(self._current_round["results"]),
            "events": self._current_round["events"],
            "transitions": dict(self._current_round["transitions"]),
        }
        self.round_history.append(summary)
        self._current_round = None
        return summary

    def should_skip_block(self, lid) -> bool:
        if lid is None:
            return False

        block_info = self.block_states.get(str(lid), {})
        state = block_info.get("state")
        block_round = block_info.get("round", 0)
        should_skip = state in self.STABLE_STATES
        if should_skip and self._current_round is not None and block_round >= self._current_round["round"]:
            return False
        if should_skip and self._current_round is not None:
            self._current_round["skipped_optimal"] += 1
        return should_skip

    def can_rewrite_block(self, lid) -> bool:
        if lid is None:
            return False
        return bool(self.block_states.get(str(lid), {}).get("meta", {}).get("rewrite_allowed", True))

    def should_disable_rewrite_globally(self) -> bool:
        return bool(self.document_profile.get("disable_global_rewrite", False))

    def is_conservative_rewrite_mode(self) -> bool:
        return bool(self.document_profile.get("conservative_rewrite_mode", False))

    def get_block_state(self, lid) -> str:
        if lid is None:
            return "REGISTERED"
        return self.block_states.get(str(lid), {}).get("state", "REGISTERED")

    def update_block_state(
        self,
        lid,
        status: str,
        round_index: int = 0,
        reason: str = "",
        metadata: Optional[Dict] = None,
        force: bool = False,
    ) -> bool:
        if lid is None or status not in self.VALID_STATES:
            return False

        key = str(lid)
        block_info = self.block_states.setdefault(
            key,
            {
                "state": "REGISTERED",
                "round": 0,
                "reason": "registered",
                "history": [],
                "meta": {},
                "m0": {},
            },
        )
        self._ensure_block_attribution(key)

        old_state = block_info.get("state", "REGISTERED")
        allowed = status in self.ALLOWED_TRANSITIONS.get(old_state, set())
        if not (allowed or force):
            self.transition_log.append(
                {
                    "lid": lid,
                    "round": round_index,
                    "from": old_state,
                    "to": status,
                    "reason": reason,
                    "metadata": deepcopy(metadata) if metadata else {},
                    "accepted": False,
                }
            )
            return False

        block_info["state"] = status
        block_info["round"] = round_index
        block_info["reason"] = reason
        if metadata:
            block_info["meta"].update(metadata)
        block_info["history"].append(
            {
                "round": round_index,
                "from": old_state,
                "state": status,
                "reason": reason,
                "metadata": deepcopy(metadata) if metadata else {},
            }
        )

        transition_event = {
            "lid": lid,
            "round": round_index,
            "from": old_state,
            "to": status,
            "reason": reason,
            "metadata": deepcopy(metadata) if metadata else {},
            "accepted": True,
        }
        self.transition_log.append(transition_event)
        self.block_attribution[key]["transitions"].append(deepcopy(transition_event))

        if metadata:
            bucket = metadata.get("m0_bucket") or metadata.get("bucket")
            if bucket:
                block_info["meta"]["bucket"] = bucket

        if self._current_round is not None:
            self._current_round["processed_lids"].add(key)
            self._current_round["results"][status] += 1
            self._current_round["transitions"][f"{old_state}->{status}"] += 1

        return True

    def decide_feedback_path(
        self,
        issue: str,
        is_title: bool = False,
        overflow_px: float = 0.0,
        is_final_round: bool = False,
    ) -> Dict:
        if issue == "OPTIMAL":
            return {"decision": "skip", "actions": []}

        if is_title:
            return {
                "decision": "micro_tune",
                "rewrite_mode": None,
                "actions": ["CharSpacing"],
            }

        if issue == "OVERFLOW":
            if is_final_round:
                return {
                    "decision": "force_fit",
                    "rewrite_mode": None,
                    "actions": ["CharSpacing"],
                }
            return {
                "decision": "rewrite_and_tune",
                "rewrite_mode": "shorten",
                "actions": ["Rewriting", "CharSpacing"],
                "overflow_px": overflow_px,
            }

        if issue == "UNDERFLOW":
            return {
                "decision": "rewrite_and_tune",
                "rewrite_mode": "lengthen",
                "actions": ["Rewriting", "CharSpacing"],
            }

        return {"decision": "observe", "rewrite_mode": None, "actions": []}

    def record_reflow_evaluation(
        self,
        lid,
        round_index: int,
        issue: str,
        decision: str,
        state_before: str,
        state_after: str,
        planned_actions: Optional[List[str]] = None,
        applied_actions: Optional[List[str]] = None,
        metrics_before: Optional[Dict] = None,
        metrics_after: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        payload = {
            "issue": issue,
            "decision": decision,
            "planned_actions": planned_actions or [],
            "applied_actions": applied_actions or [],
            "metrics_before": deepcopy(metrics_before) if metrics_before else {},
            "metrics_after": deepcopy(metrics_after) if metrics_after else {},
        }
        if metadata:
            payload.update(deepcopy(metadata))

        self._record_attribution_event(
            lid,
            round_index,
            stage="reflow",
            event_type="evaluation",
            state_before=state_before,
            state_after=state_after,
            planned_actions=planned_actions or [],
            applied_actions=applied_actions or [],
            metadata=payload,
        )
        self.issue_counter[issue] += 1

    def record_rewrite_result(
        self,
        lid,
        round_index: int,
        mode: str,
        before_text: str,
        after_text: str,
        updated_indices: List[int],
        metadata: Optional[Dict] = None,
    ) -> None:
        payload = {
            "rewrite_mode": mode,
            "before_chars": len(before_text),
            "after_chars": len(after_text),
            "char_delta": len(after_text) - len(before_text),
            "updated_indices": deepcopy(updated_indices),
        }
        if metadata:
            payload.update(deepcopy(metadata))

        self._record_attribution_event(
            lid,
            round_index,
            stage="rewrite",
            event_type="rewrite_apply",
            state_before="REWRITE_SCHEDULED",
            state_after="PENDING",
            planned_actions=["Rewriting"],
            applied_actions=["Rewriting"],
            metadata=payload,
        )

    def record_render_result(
        self,
        lid,
        item_index: int,
        final_style: Dict,
        metrics: Dict,
        metadata: Optional[Dict] = None,
    ) -> None:
        payload = {
            "item_index": item_index,
            "final_style": deepcopy(final_style),
            "render_metrics": deepcopy(metrics),
        }
        if metadata:
            payload.update(deepcopy(metadata))

        self._record_attribution_event(
            lid,
            round_index=0,
            stage="render",
            event_type="render_finalize",
            state_before=self.get_block_state(lid),
            state_after=self.get_block_state(lid),
            planned_actions=[],
            applied_actions=[],
            metadata=payload,
        )

    def record_action(
        self,
        lid,
        round_index: int,
        issue: str,
        actions: List[str],
        result: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Backward-compatible wrapper. Keeps older call sites working while
        routing everything through the new attribution model.
        """
        payload = deepcopy(metadata) if metadata else {}
        payload["issue"] = issue
        payload["result"] = result
        self._record_attribution_event(
            lid,
            round_index,
            stage="legacy",
            event_type="action_summary",
            state_before=self.get_block_state(lid),
            state_after=self.get_block_state(lid),
            planned_actions=actions,
            applied_actions=actions,
            metadata=payload,
        )
        self.issue_counter[issue] += 1

    def export_summary(self) -> Dict:
        return {
            "document_profile": deepcopy(self.document_profile),
            "global_plan": deepcopy(self.global_plan),
            "block_states": deepcopy(self.block_states),
            "action_stats": dict(self.action_counter),
            "issue_stats": dict(self.issue_counter),
            "round_history": deepcopy(self.round_history),
            "action_log": deepcopy(self.action_log),
            "transition_log": deepcopy(self.transition_log),
            "block_attribution": deepcopy(self.block_attribution),
            "m0_probe_log": deepcopy(self.m0_probe_log),
        }

    def _record_attribution_event(
        self,
        lid,
        round_index: int,
        stage: str,
        event_type: str,
        state_before: str,
        state_after: str,
        planned_actions: List[str],
        applied_actions: List[str],
        metadata: Optional[Dict] = None,
    ) -> None:
        key = str(lid) if lid is not None else "document"
        self._ensure_block_attribution(key)

        normalized_planned = self._unique_nonempty(planned_actions)
        normalized_applied = self._unique_nonempty(applied_actions)
        event = {
            "lid": lid,
            "round": round_index,
            "stage": stage,
            "event_type": event_type,
            "state_before": state_before,
            "state_after": state_after,
            "planned_actions": normalized_planned,
            "applied_actions": normalized_applied,
            "metadata": deepcopy(metadata) if metadata else {},
        }
        self.block_attribution[key]["events"].append(deepcopy(event))
        self.action_log.append(deepcopy(event))

        for action in normalized_applied:
            self.action_counter[action] += 1
        if self._current_round is not None:
            self._current_round["processed_lids"].add(key)
            self._current_round["events"] += 1
            for action in normalized_applied:
                self._current_round["actions"][action] += 1
            issue = (metadata or {}).get("issue")
            if issue:
                self._current_round["issues"][issue] += 1

    def _ensure_block_attribution(self, key: str) -> None:
        if key not in self.block_attribution:
            self.block_attribution[key] = {
                "events": [],
                "transitions": [],
            }

    def _analyze_document(self, data: List[Dict], pdf_path: str) -> Dict:
        page_count = 0
        page_width = 0.0
        page_height = 0.0

        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            if page_count > 0:
                first_rect = doc[0].rect
                page_width = first_rect.width
                page_height = first_rect.height
            doc.close()
        except Exception:
            pass

        text_items = [
            item
            for item in data
            if item.get("type", "").lower() not in {"image", "table", "figure", "line", "rect", "curve"}
            and item.get("text", "").strip()
        ]
        text_lengths = [len(item.get("text", "").strip()) for item in text_items if item.get("text", "").strip()]
        bbox_areas = []
        source_densities = []

        for item in text_items:
            bbox = item.get("bbox", [])
            if len(bbox) != 4:
                continue
            width = max(0.0, bbox[2] - bbox[0])
            height = max(0.0, bbox[3] - bbox[1])
            area = width * height
            if area <= 0:
                continue
            bbox_areas.append(area)
            text_len = max(1, len(item.get("text", "").strip()))
            source_densities.append(area / text_len)

        return {
            "pdf_path": pdf_path,
            "page_count": page_count,
            "page_width": page_width,
            "page_height": page_height,
            "block_count": len(data),
            "text_block_count": len(text_items),
            "avg_text_length": round(statistics.mean(text_lengths), 2) if text_lengths else 0.0,
            "median_source_density": round(statistics.median(source_densities), 4) if source_densities else 0.0,
            "avg_bbox_area": round(statistics.mean(bbox_areas), 2) if bbox_areas else 0.0,
        }

    def _extract_golden_font_size(self, global_styles: Dict) -> Optional[float]:
        if not global_styles:
            return None

        body_style = global_styles.get("body", {})
        if body_style.get("size") is not None:
            return body_style["size"]

        sizes = [
            style.get("size")
            for style in global_styles.values()
            if isinstance(style, dict) and style.get("size") is not None
        ]
        if not sizes:
            return None
        return self._round_half(
            statistics.mode(sizes) if len(set(sizes)) < len(sizes) else statistics.median(sizes)
        )

    def _extract_golden_line_spacing(self, global_styles: Dict) -> Optional[float]:
        if not global_styles:
            return None

        body_style = global_styles.get("body", {})
        if body_style.get("line") is not None:
            return body_style["line"]

        lines = [
            style.get("line")
            for style in global_styles.values()
            if isinstance(style, dict) and style.get("line") is not None
        ]
        if not lines:
            return None
        return self._round_to_step(
            statistics.mode(lines) if len(set(lines)) < len(lines) else statistics.median(lines),
            0.05,
        )

    def _math_segment_count(self, text: str) -> int:
        if not text:
            return 0
        inline_pairs = text.count("$") // 2
        display_pairs = text.count("\\[") + text.count("\\(")
        return inline_pairs + display_pairs

    def _latex_command_count(self, text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r"\\[A-Za-z]+", text))

    def _is_text_like_segment(self, item: Dict) -> bool:
        raw_type = (item.get("type") or "").lower()
        detected_type = (item.get("detected_type") or raw_type).lower()
        if any(token in raw_type or token in detected_type for token in self.REWRITE_PRESERVE_TYPES):
            return False
        text_like_raw = {"", "text", "abstract", "paragraph"}
        text_like_detected = {"", "text", "body", "paragraph", "abstract"}
        return raw_type in text_like_raw and detected_type in text_like_detected

    def _is_safe_multibox_body_candidate(
        self,
        items: List[Dict],
        bucket: str,
        compact_text: str,
        math_segments: int,
        latex_commands: int,
        theorem_like: bool,
        compact_box: bool,
    ) -> bool:
        min_chars = int(self.style_policy.get("safe_prose_rewrite_min_chars", 60))
        if (
            bucket != "body"
            or len(items) <= 1
            or len(compact_text) < min_chars
            or math_segments != 0
            or latex_commands != 0
            or theorem_like
            or compact_box
        ):
            return False
        if len(items) > 3:
            return False
        return all(self._is_text_like_segment(item) for item in items)

    def _theorem_like_text(self, text: str) -> bool:
        if not text:
            return False
        return bool(
            re.search(
                r"\b(theorem|lemma|proof|corollary|proposition|definition|remark|example|claim)\b",
                text,
                re.IGNORECASE,
            )
        )

    def _annotate_document_risk(self, data: List[Dict]) -> None:
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for item in data:
            lid = item.get("logical_para_id")
            if lid is None:
                continue
            groups[str(lid)].append(item)

        logical_blocks = []
        math_like_blocks = 0
        theorem_like_blocks = 0
        multi_box_blocks = 0
        safe_body_blocks = 0

        for items in groups.values():
            items = sorted(items, key=lambda x: (x.get("page_idx", x.get("page", 0)), x.get("bbox", [0, 0, 0, 0])[1]))
            first = items[0]
            raw_type = (first.get("type") or "").lower()
            if raw_type in {"image", "table", "figure", "line", "rect", "curve", "equation", "code"}:
                continue
            text = first.get("context") or " ".join((item.get("text") or "") for item in items)
            compact_text = re.sub(r"\s+", "", text or "")
            bbox = first.get("bbox", [])
            box_height = max(0.0, float(bbox[3]) - float(bbox[1])) if len(bbox) == 4 else 0.0
            math_segments = self._math_segment_count(text)
            latex_commands = self._latex_command_count(text)
            theorem_like = self._theorem_like_text(text)
            compact_box = box_height > 0 and box_height <= 22
            text_like = raw_type in {"text", "abstract", ""}
            logical_blocks.append(items)
            if len(items) > 1:
                multi_box_blocks += 1
            if math_segments > 0 or latex_commands >= 2:
                math_like_blocks += 1
            if theorem_like:
                theorem_like_blocks += 1
            if self._is_safe_prose_body_candidate(
                bucket="body" if text_like else "",
                compact_text=compact_text,
                math_segments=math_segments,
                latex_commands=latex_commands,
                theorem_like=theorem_like,
                multi_box=len(items) > 1,
                compact_box=compact_box,
                items=items,
            ):
                safe_body_blocks += 1

        logical_count = max(1, len(logical_blocks))
        math_ratio = math_like_blocks / logical_count
        theorem_ratio = theorem_like_blocks / logical_count
        multibox_ratio = multi_box_blocks / logical_count
        safe_body_ratio = safe_body_blocks / logical_count

        conservative_mode = (
            math_ratio >= float(self.style_policy.get("rewrite_conservative_doc_math_ratio", 0.18))
            or theorem_ratio >= 0.06
            or (math_ratio >= 0.12 and multibox_ratio >= 0.08)
        )
        disable_global_rewrite = (
            math_ratio >= float(self.style_policy.get("rewrite_disable_doc_math_ratio", 0.32))
            or (math_ratio >= 0.18 and multibox_ratio >= float(self.style_policy.get("rewrite_disable_doc_multibox_ratio", 0.12)))
        )
        if disable_global_rewrite and (
            safe_body_blocks >= int(self.style_policy.get("rewrite_disable_safe_body_min_blocks", 12))
            or safe_body_ratio >= float(self.style_policy.get("rewrite_disable_safe_body_ratio", 0.12))
        ):
            disable_global_rewrite = False
            conservative_mode = True

        self.document_profile.update(
            {
                "logical_block_count": logical_count,
                "math_like_block_count": math_like_blocks,
                "math_like_block_ratio": round(math_ratio, 4),
                "theorem_like_block_count": theorem_like_blocks,
                "theorem_like_block_ratio": round(theorem_ratio, 4),
                "multi_box_block_count": multi_box_blocks,
                "multi_box_block_ratio": round(multibox_ratio, 4),
                "safe_body_block_count": safe_body_blocks,
                "safe_body_block_ratio": round(safe_body_ratio, 4),
                "conservative_rewrite_mode": conservative_mode,
                "disable_global_rewrite": disable_global_rewrite,
            }
        )
        self.global_plan["rewrite_policy"] = {
            "conservative_rewrite_mode": conservative_mode,
            "disable_global_rewrite": disable_global_rewrite,
            "math_like_block_ratio": round(math_ratio, 4),
            "theorem_like_block_ratio": round(theorem_ratio, 4),
            "multi_box_block_ratio": round(multibox_ratio, 4),
            "safe_body_block_count": safe_body_blocks,
            "safe_body_block_ratio": round(safe_body_ratio, 4),
        }

    def _annotate_block_risks(self, data: List[Dict]) -> None:
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for item in data:
            lid = item.get("logical_para_id")
            if lid is None:
                continue
            groups[str(lid)].append(item)

        conservative_mode = self.is_conservative_rewrite_mode()
        disable_global_rewrite = self.should_disable_rewrite_globally()
        rewrite_counts = Counter()

        for key, items in groups.items():
            items = sorted(items, key=lambda x: (x.get("page_idx", x.get("page", 0)), x.get("bbox", [0, 0, 0, 0])[1]))
            first = items[0]
            text = first.get("context") or " ".join((item.get("text") or "") for item in items)
            compact_text = re.sub(r"\s+", "", text or "")
            raw_type = (first.get("type") or "").lower()
            detected_type = (first.get("detected_type") or raw_type).lower()
            bucket = self._resolve_bucket(first, first.get("golden_style"))
            block_info = self.block_states.setdefault(
                key,
                {"state": "REGISTERED", "round": 0, "reason": "registered", "history": [], "meta": {}, "m0": {}},
            )

            bbox = first.get("bbox", [])
            box_height = max(0.0, float(bbox[3]) - float(bbox[1])) if len(bbox) == 4 else 0.0
            math_segments = self._math_segment_count(text)
            latex_commands = self._latex_command_count(text)
            theorem_like = self._theorem_like_text(text)
            multi_box = len(items) > 1
            compact_box = box_height > 0 and box_height <= 22
            long_text = len(compact_text) >= int(self.style_policy.get("rewrite_block_max_chars", 220))
            safe_prose_body = self._is_safe_prose_body_candidate(
                bucket=bucket,
                compact_text=compact_text,
                math_segments=math_segments,
                latex_commands=latex_commands,
                theorem_like=theorem_like,
                multi_box=multi_box,
                compact_box=compact_box,
                items=items,
            )
            safe_multibox_body = multi_box and self._is_safe_multibox_body_candidate(
                items=items,
                bucket=bucket,
                compact_text=compact_text,
                math_segments=math_segments,
                latex_commands=latex_commands,
                theorem_like=theorem_like,
                compact_box=compact_box,
            )

            risk_flags = []
            if multi_box:
                risk_flags.append("multi_box")
            if safe_multibox_body:
                risk_flags.append("safe_multibox_body")
            if math_segments > 0:
                risk_flags.append("contains_math")
            if latex_commands > 0:
                risk_flags.append("contains_latex_commands")
            if theorem_like:
                risk_flags.append("theorem_like")
            if compact_box:
                risk_flags.append("compact_box")
            if long_text:
                risk_flags.append("long_text")
            if safe_prose_body:
                risk_flags.append("safe_prose_body")
            if bucket in {"title", "section_header", "caption", "footnote", "header", "footer"}:
                risk_flags.append(f"bucket_{bucket}")
            if raw_type in {"aside_text", "page_number"}:
                risk_flags.append(f"type_{raw_type}")

            rewrite_allowed = self._is_rewrite_candidate_block(bucket, raw_type, detected_type)
            high_risk_rewrite = multi_box or math_segments > 0 or latex_commands > 0 or theorem_like or compact_box
            if disable_global_rewrite:
                risk_flags.append("doc_global_rewrite_disabled")
                if safe_prose_body:
                    risk_flags.append("safe_prose_rewrite_override")
            if conservative_mode and high_risk_rewrite:
                risk_flags.append("doc_conservative_mode")

            risk_level = "low"
            if any(flag in risk_flags for flag in ("contains_math", "contains_latex_commands", "theorem_like", "multi_box")):
                risk_level = "high"
            elif risk_flags:
                risk_level = "medium"

            block_info["meta"].update(
                {
                    "bucket": bucket,
                    "risk_flags": sorted(set(risk_flags)),
                    "risk_level": risk_level,
                    "rewrite_allowed": rewrite_allowed,
                    "math_segments": math_segments,
                    "latex_commands": latex_commands,
                    "multi_box": multi_box,
                    "compact_box": compact_box,
                    "long_text": long_text,
                    "theorem_like": theorem_like,
                    "safe_prose_rewrite": safe_prose_body,
                    "safe_multibox_body": safe_multibox_body,
                    "multi_box_segment_count": len(items),
                    "detected_type": detected_type,
                }
            )
            rewrite_counts["allowed" if rewrite_allowed else "blocked"] += 1

        self.global_plan.setdefault("rewrite_policy", {})
        self.global_plan["rewrite_policy"].update(
            {
                "rewrite_allowed_blocks": rewrite_counts.get("allowed", 0),
                "rewrite_blocked_blocks": rewrite_counts.get("blocked", 0),
            }
        )

    def _is_safe_prose_body_candidate(
        self,
        bucket: str,
        compact_text: str,
        math_segments: int,
        latex_commands: int,
        theorem_like: bool,
        multi_box: bool,
        compact_box: bool,
        items: List[Dict] = None,
    ) -> bool:
        min_chars = int(self.style_policy.get("safe_prose_rewrite_min_chars", 60))
        if (
            bucket == "body"
            and len(compact_text) >= min_chars
            and math_segments == 0
            and latex_commands == 0
            and not theorem_like
            and not compact_box
        ):
            if not multi_box:
                return True
            return self._is_safe_multibox_body_candidate(
                items=items or [],
                bucket=bucket,
                compact_text=compact_text,
                math_segments=math_segments,
                latex_commands=latex_commands,
                theorem_like=theorem_like,
                compact_box=compact_box,
            )
        return False

    def _is_rewrite_candidate_block(self, bucket: str, raw_type: str, detected_type: str) -> bool:
        raw = (raw_type or "").lower()
        detected = (detected_type or raw).lower()
        if any(token in raw or token in detected for token in self.REWRITE_PRESERVE_TYPES):
            return False
        if bucket in {"title", "section_header", "header", "footer"}:
            return False
        return True

    def _is_dense_probe_block(
        self,
        bucket: str,
        bbox: List[float],
        text: str,
        base_size: float,
        base_line: float,
    ) -> bool:
        if bucket != "body" or len(bbox) != 4:
            return False

        box_height = max(0.0, float(bbox[3]) - float(bbox[1]))
        compact_chars = len(text.replace(" ", "").replace("\n", ""))
        math_segments = self._math_segment_count(text)
        nominal_line_h = max(base_size * base_line, 1.0)
        visible_lines = box_height / nominal_line_h if nominal_line_h else 0.0

        return compact_chars >= 70 or math_segments >= 2 or visible_lines <= 1.6

    def _resolve_bucket(self, item: Dict, style: Optional[Dict] = None) -> str:
        detected = (item.get("detected_type") or item.get("type") or "").lower()
        if detected in self.BUCKET_ALIASES:
            bucket = self.BUCKET_ALIASES[detected]
        elif "section_header" in detected:
            bucket = "section_header"
        elif "title" in detected:
            bucket = "title"
        elif "caption" in detected:
            bucket = "caption"
        elif "footnote" in detected:
            bucket = "footnote"
        elif "header" in detected:
            bucket = "header"
        elif "footer" in detected:
            bucket = "footer"
        else:
            font_key = (style or {}).get("font_key", "body")
            bucket = font_key if font_key in {"title", "heading", "caption", "body"} else "body"
            if bucket == "heading":
                bucket = "section_header"
        lid = item.get("logical_para_id")
        if lid is not None:
            key = str(lid)
            if key in self.block_states:
                self.block_states[key]["meta"]["bucket"] = bucket
        return bucket

    def _freeze_bucket_styles(self, bucket_probes: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
        frozen = {}
        for bucket, values in bucket_probes.items():
            bucket_style = {}
            if values.get("size"):
                bucket_style["size"] = self._freeze_size_series_for_bucket(bucket, values["size"])
            if values.get("line"):
                bucket_style["line"] = self._freeze_numeric_series(values["line"], step=0.05)
            if values.get("char"):
                bucket_style["char"] = self._freeze_numeric_series(values["char"], step=0.005)
            if bucket_style:
                frozen[bucket] = bucket_style
        return frozen

    def _apply_frozen_bucket_styles(self, styles: Dict, frozen_bucket_styles: Dict[str, Dict[str, float]]) -> Dict:
        frozen_styles = deepcopy(styles)
        for style_name, style in frozen_styles.items():
            if not isinstance(style, dict):
                continue
            probe_item = {"type": style_name}
            bucket = self._resolve_bucket(probe_item, style)
            if bucket in frozen_bucket_styles:
                if frozen_bucket_styles[bucket].get("size") is not None:
                    style["size"] = frozen_bucket_styles[bucket]["size"]
                if frozen_bucket_styles[bucket].get("line") is not None:
                    style["line"] = frozen_bucket_styles[bucket]["line"]
                if frozen_bucket_styles[bucket].get("char") is not None:
                    style["char"] = frozen_bucket_styles[bucket]["char"]
        return frozen_styles

    def _probe_block_style(self, item: Dict, bucket: str, style: Dict) -> Dict:
        bbox = item.get("bbox", [])
        text = item.get("translated", "").strip() or item.get("text", "").strip()
        if len(bbox) != 4 or not text:
            return {
                "size": style.get("size"),
                "line": style.get("line"),
                "char": style.get("char"),
                "fill_ratio": 0.0,
                "overflow_px": 0.0,
                "score": 9999.0,
            }

        tool = self._get_layout_tool()
        base_size = float(style.get("size", 10.5))
        base_line = float(style.get("line", 1.35))
        base_char = float(style.get("char", 0.05))
        is_dense = self._is_dense_probe_block(bucket, bbox, text, base_size, base_line)
        target_fill = self._target_fill_for_bucket(bucket, is_dense=is_dense)
        min_size = max(
            6.0,
            base_size * (0.85 if bucket in {"title", "section_header"} else (0.65 if is_dense else 0.75)),
        )
        if bucket in {"title", "section_header", "body", "caption", "footnote", "header", "footer"}:
            max_size = base_size
        else:
            max_size = min(18.0, base_size * 1.1)
        line_candidates = self._build_line_candidates(base_line, bucket)
        char_candidates = self._build_char_candidates(base_char)
        box_height = max(0.0, float(bbox[3]) - float(bbox[1]))
        safe_box_height = max(1.0, box_height * self._safe_box_ratio_for_probe(bucket, is_dense=is_dense))

        size_candidates = []
        current = self._round_half(min_size)
        while current <= max_size + 1e-6:
            size_candidates.append(self._round_half(current))
            current += 0.5
        if self._round_half(base_size) not in size_candidates:
            size_candidates.append(self._round_half(base_size))
        size_candidates = sorted(set(size_candidates))

        best = None
        for size in size_candidates:
            for line in line_candidates:
                for char in char_candidates:
                    candidate_style = deepcopy(style)
                    candidate_style["size"] = size
                    candidate_style["line"] = line
                    candidate_style["char"] = char
                    metrics = tool.simulate_layout_metrics(bbox, text, candidate_style)
                    needed_height = float(metrics.get("needed_height", 0.0))
                    safe_fill_ratio = needed_height / safe_box_height if safe_box_height > 0 else 0.0
                    soft_overflow = needed_height > safe_box_height
                    overflow_px = max(
                        0.0,
                        needed_height - safe_box_height,
                    )
                    overflow_penalty = 1500.0 if metrics.get("is_overflow") else (450.0 if soft_overflow else 0.0)
                    size_up_penalty = max(0.0, size - base_size) * (30.0 if bucket == "body" else 10.0)
                    line_up_penalty = 0.0
                    if bucket == "body" and self.document_profile.get("layout_mode", "single_col") == "single_col":
                        line_up_penalty = max(0.0, line - base_line) * 55.0
                    score = (
                        overflow_penalty
                        + abs(safe_fill_ratio - target_fill) * 120.0
                        + abs(line - base_line) * 35.0
                        + abs(char - base_char) * 400.0
                        + size_up_penalty
                        + line_up_penalty
                        - size * 0.05
                    )
                    record = {
                        "size": self._round_half(size),
                        "line": self._round_to_step(line, 0.05),
                        "char": self._round_to_step(char, 0.005),
                        "fill_ratio": round(safe_fill_ratio, 4),
                        "overflow_px": round(overflow_px, 4),
                        "score": round(score, 4),
                    }
                    if best is None or record["score"] < best["score"]:
                        best = record

        return best or {
            "size": self._round_half(base_size),
            "line": self._round_to_step(base_line, 0.05),
            "char": self._round_to_step(base_char, 0.005),
            "fill_ratio": 0.0,
            "overflow_px": 0.0,
            "score": 9999.0,
        }

    def _target_fill_for_bucket(self, bucket: str, is_dense: bool = False) -> float:
        if bucket == "body":
            configured = (
                self.style_policy["dense_body_probe_safe_fill"]
                if is_dense
                else self.style_policy["body_probe_safe_fill"]
            )
            target = max(configured, 0.90 if is_dense else 0.94)
            density = float(self.document_profile.get("median_source_density", 0.0) or 0.0)
            if density and density <= 58.0:
                target += 0.008
            if density and density <= 50.0:
                target += 0.01
            if density and density <= 42.0:
                target += 0.008
            if self.document_profile.get("layout_mode", "single_col") == "double_col":
                target += 0.004
            if self.document_profile.get("avg_text_length", 0.0) >= 320.0:
                target += 0.004
            return min(0.992, target)

        target_map = {
            "title": 0.86,
            "section_header": 0.90,
            "caption": 0.88,
            "footnote": 0.86,
            "header": 0.84,
            "footer": 0.84,
        }
        return target_map.get(bucket, self.style_policy["other_probe_safe_fill"])

    def _get_layout_tool(self):
        if self._layout_tool is None:
            from tools.pdf_reflow_tool import PDFReflowTool

            self._layout_tool = PDFReflowTool(lang=self.target_lang)
        return self._layout_tool

    def _round_half(self, value: float) -> float:
        return round(float(value) * 2) / 2

    def _round_to_step(self, value: float, step: float) -> float:
        return round(float(value) / step) * step

    def _freeze_numeric_series(self, values: List[float], step: float) -> float:
        rounded = [self._round_to_step(value, step) for value in values if value is not None]
        if not rounded:
            return None
        counts = Counter(rounded)
        top_count = counts.most_common(1)[0][1]
        top_values = sorted(value for value, count in counts.items() if count == top_count)
        if len(rounded) >= 4:
            frozen_value = statistics.median(top_values) if len(top_values) > 1 else top_values[0]
        else:
            frozen_value = statistics.median(rounded)
        return self._round_to_step(frozen_value, step)

    def _freeze_size_series_for_bucket(self, bucket: str, values: List[float]) -> Optional[float]:
        rounded = [self._round_half(value) for value in values if value is not None]
        if not rounded:
            return None

        if bucket != "body":
            return self._freeze_numeric_series(rounded, step=0.5)

        ordered = sorted(rounded)
        if len(ordered) == 1:
            return ordered[0]
        if len(ordered) < 6:
            return self._round_half(statistics.median_low(ordered))

        quantile = min(0.5, max(0.1, float(self.style_policy.get("body_freeze_quantile", 0.35))))
        spread = ordered[-1] - ordered[0]
        if spread >= 2.5:
            quantile = min(
                quantile,
                min(
                    0.25,
                    max(0.1, float(self.style_policy.get("body_freeze_quantile_spread2", 0.15))),
                ),
            )
        elif spread >= 1.5:
            quantile = min(
                quantile,
                min(
                    0.3,
                    max(0.1, float(self.style_policy.get("body_freeze_quantile_spread1", 0.2))),
                ),
            )
        position = (len(ordered) - 1) * quantile
        lower_idx = int(position)
        upper_idx = min(len(ordered) - 1, lower_idx + 1)
        lower = ordered[lower_idx]
        upper = ordered[upper_idx]
        interpolated = lower + (upper - lower) * (position - lower_idx)
        return self._round_half(interpolated)

    def _safe_box_ratio_for_probe(self, bucket: str, is_dense: bool = False) -> float:
        if bucket == "body":
            return (
                self.style_policy["dense_probe_safe_box_ratio"]
                if is_dense
                else self.style_policy["body_probe_safe_box_ratio"]
            )
        if bucket in {"title", "section_header"}:
            return 0.96
        return self.style_policy["other_probe_safe_box_ratio"]

    def _build_line_candidates(self, base_line: float, bucket: str) -> List[float]:
        flex = 0.05 if bucket in {"title", "section_header"} else 0.1
        candidates = {
            self._round_to_step(max(1.1, base_line - flex), 0.05),
            self._round_to_step(max(1.1, base_line - 0.05), 0.05),
            self._round_to_step(base_line, 0.05),
            self._round_to_step(base_line + 0.05, 0.05),
        }
        if bucket == "body":
            layout_mode = self.document_profile.get("layout_mode", "single_col")
            candidates.add(self._round_to_step(max(1.1, base_line - 0.15), 0.05))
            if layout_mode == "single_col":
                candidates.add(self._round_to_step(max(1.1, base_line - 0.20), 0.05))
                candidates.add(self._round_to_step(base_line + 0.10, 0.05))
            else:
                candidates.add(self._round_to_step(base_line + 0.10, 0.05))
                candidates.add(self._round_to_step(base_line + 0.15, 0.05))
        return sorted(candidates)

    def _build_char_candidates(self, base_char: float) -> List[float]:
        abs_min = self.style_policy["char_spacing_abs_min"]
        abs_max = self.style_policy["char_spacing_abs_max"]
        delta = self.style_policy["char_spacing_max_delta_from_bucket"]
        candidates = {
            self._round_to_step(min(abs_max, max(abs_min, base_char - delta)), 0.005),
            self._round_to_step(min(abs_max, max(abs_min, base_char - delta / 2)), 0.005),
            self._round_to_step(min(abs_max, max(abs_min, base_char)), 0.005),
            self._round_to_step(min(abs_max, max(abs_min, base_char + delta / 2)), 0.005),
            self._round_to_step(min(abs_max, max(abs_min, base_char + delta)), 0.005),
        }
        candidates.add(self._round_to_step(min(abs_max, max(abs_min, base_char + delta + 0.005)), 0.005))
        return sorted(candidates)

    def _build_char_spacing_guardrails(
        self,
        frozen_bucket_styles: Dict[str, Dict[str, float]],
        fallback_styles: Optional[Dict] = None,
    ) -> Dict[str, float]:
        chars = [
            style["char"]
            for style in frozen_bucket_styles.values()
            if isinstance(style, dict) and style.get("char") is not None
        ]
        if not chars and fallback_styles:
            chars = [
                style.get("char")
                for style in fallback_styles.values()
                if isinstance(style, dict) and style.get("char") is not None
            ]
        abs_min = self.style_policy["char_spacing_abs_min"]
        abs_max = self.style_policy["char_spacing_abs_max"]
        max_spread = self.style_policy["char_spacing_max_global_spread"]
        max_delta = self.style_policy["char_spacing_max_delta_from_bucket"]
        center = statistics.median(chars) if chars else 0.05
        global_min = max(abs_min, center - max_spread / 2)
        global_max = min(abs_max, center + max_spread / 2)
        if global_max < global_min:
            global_min, global_max = abs_min, abs_max
        return {
            "absolute_min": abs_min,
            "absolute_max": abs_max,
            "global_min": self._round_to_step(global_min, 0.005),
            "global_max": self._round_to_step(global_max, 0.005),
            "center": self._round_to_step(center, 0.005),
            "max_delta_from_bucket": max_delta,
            "max_global_spread": max_spread,
        }

    def _clamp_char_with_guardrails(self, value: float, guardrails: Dict[str, float]) -> float:
        clamped = min(guardrails["global_max"], max(guardrails["global_min"], value))
        return self._round_to_step(clamped, 0.005)

    def _clamp_frozen_style_chars(self, styles: Dict, guardrails: Dict[str, float]) -> Dict:
        clamped_styles = deepcopy(styles)
        for style in clamped_styles.values():
            if isinstance(style, dict) and style.get("char") is not None:
                style["char"] = self._clamp_char_with_guardrails(style["char"], guardrails)
        return clamped_styles

    def _unique_nonempty(self, values: List[str]) -> List[str]:
        result = []
        for value in values:
            if value and value not in result:
                result.append(value)
        return result
