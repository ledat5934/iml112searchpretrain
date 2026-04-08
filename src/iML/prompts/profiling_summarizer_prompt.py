import json
from typing import Dict, Any

from .base_prompt import BasePrompt


class ProfilingSummarizerPrompt(BasePrompt):
    """Prompt to condense verbose profiling into compact JSON signals for pipeline."""

    def default_template(self) -> str:
        tmpl = (
            """
You are a senior ML data analyst. Read the dataset description and the RAW profiling result below.
Your job is to produce a COMPACT, ACTIONABLE JSON summary for downstream preprocessing and modeling.

Focus on clear, minimal signals, but include high-signal schema facts (column names, dtypes, join keys, file roles)
when they materially prevent hallucinations in downstream code generation. Avoid dumping large arrays.

MUST OUTPUT ONLY VALID JSON with these keys:
{{
    "dataset_name": str,
    "task_type_hint": str | null,
    "modalities": [str],  // e.g., ["tabular"], ["audio"], ["image","audio"], ...
    "key_files": [
        {{
          "rel_path": str,
          "role": "submission|labels|folds|id_mapping|class_list|train_table|test_table|unknown",
          "columns": [str] | null,
          "dtypes": object | null,
          "notes": str | null
        }}
    ],
    "join_keys": [
        {{"key": str, "files": [str], "confidence": "low|medium|high", "notes": str}}
    ],
    "data_split_hint": {{
        "train_files": [str],
        "test_files": [str],
        "sample_submission_file": str | null,
        "fold_files": [str],
        "notes": [str]
    }},
    "label_analysis": {{
        "has_label_column": bool,
        "label_column": str | null,
        "has_missing_labels": bool | null,
        "num_classes": int | null,
        "class_distribution_imbalance": "none|mild|moderate|severe|null",
        "notes": str
    }},
    "feature_quality": {{
        "high_missing_columns": [str],
        "constant_or_near_constant_cols": [str],
        "high_cardinality_categoricals": [str],
        "date_like_cols": [str]
    }},
    "id_format_analysis": {{
        "has_file_extensions": bool,
        "detected_extensions": [str],
        "format_notes": [str]
    }},
    "critical_warnings": [str],
    "notes": [str]
}}

Rules:
- Use the provided profiling_result JSON (including llm_profiling/basic_profiles if present) to infer signals succinctly.
- If llm_profiling contains a freeform prose report (freeform_report), treat it as first-class evidence alongside structured fields.
- Prefer FACTS from schemas/relational_signals and the freeform report over guesses. If a key file exists, cite its rel_path and columns/dtypes.
- Pay special attention to split/label/mapping files; missing these causes downstream hallucinations.
- If unsure, set fields to null and explain briefly in notes.
- Do NOT output markdown fences. Output pure JSON only.

---
DESCRIPTION:
{description}

RAW_PROFILING:
```json
{profiling_compact}
```
"""
        )
        return tmpl

    def build(self, profiling_result: Dict[str, Any], description_analysis: Dict[str, Any]) -> str:
        # Compact the profiling_result before sending to LLM to reduce noise while preserving schema facts.
        summaries = profiling_result.get("summaries", {}) or {}
        profiles = profiling_result.get("profiles", {}) or {}
        id_format_analysis = profiling_result.get("id_format_analysis", {}) or {}
        basic_inventory = profiling_result.get("basic_inventory", {}) or {}
        basic_profiles = profiling_result.get("basic_profiles", []) or []
        Fmeta = profiling_result.get("Fmeta", {}) or {}

        llm_prof = (profiling_result.get("llm_profiling", {}) or {}).get("result", {}) or {}
        llm_schemas = (llm_prof.get("schemas", {}) or {}) if isinstance(llm_prof, dict) else {}
        llm_rel = (llm_prof.get("relational_signals", {}) or {}) if isinstance(llm_prof, dict) else {}
        llm_signals = (llm_prof.get("signals", {}) or {}) if isinstance(llm_prof, dict) else {}
        llm_freeform = ""
        if isinstance(llm_prof, dict):
            if llm_prof.get("format") == "text" and llm_prof.get("report"):
                llm_freeform = str(llm_prof.get("report") or "")[:12000]
            elif llm_prof.get("report") and not llm_schemas:
                llm_freeform = str(llm_prof.get("report") or "")[:12000]

        # Limit sizes to keep prompt manageable
        def _take_list(x, n: int):
            return x[:n] if isinstance(x, list) else []

        # Extract a few representative media meta entries
        media = _take_list(llm_schemas.get("media", []), 12)
        tabular = llm_schemas.get("tabular", [])
        # Prefer tabular schemas likely to be metadata (txt/csv) instead of giant tables
        tabular = [t for t in (tabular or []) if isinstance(t, dict)]
        tabular_meta_like = [t for t in tabular if str(t.get("rel_path", "")).lower().endswith((".txt", ".csv", ".tsv", ".json"))]
        tabular_compact = _take_list(tabular_meta_like, 25) or _take_list(tabular, 25)

        # Light profiles from ydata_profiling output (can be huge)
        profiles_light = {}
        for f, prof in (profiles or {}).items():
            vars_info = prof.get("variables", {}) if isinstance(prof, dict) else {}
            light_vars = {}
            # Only keep up to 40 variables per file to avoid bloat
            for i, (v, info) in enumerate(vars_info.items()):
                if i >= 40:
                    break
                light_vars[v] = {
                    "type": info.get("type"),
                    "n_unique": info.get("n_unique"),
                    "p_missing": info.get("p_missing"),
                }
            profiles_light[f] = {"variables": light_vars}

        compact = {
            "Fmeta": Fmeta,
            "basic_inventory": {
                "counts_by_ext": (basic_inventory or {}).get("counts_by_ext"),
                "examples_by_ext": (basic_inventory or {}).get("examples_by_ext"),
            },
            # Keep only a few basic_profiles entries to hint modality; details are in llm_prof
            "basic_profiles_sample": _take_list(basic_profiles, 10),
            "llm_profiling": {
                "inventory": llm_prof.get("inventory") if isinstance(llm_prof, dict) else None,
                "freeform_report": llm_freeform or None,
                "schemas": {"tabular": tabular_compact, "media": media},
                "relational_signals": {
                    "file_roles": _take_list(llm_rel.get("file_roles", []), 30),
                    "join_keys": _take_list(llm_rel.get("join_keys", []), 15),
                    "recommended_split_files": _take_list(llm_rel.get("recommended_split_files", []), 10),
                    "recommended_label_files": _take_list(llm_rel.get("recommended_label_files", []), 10),
                    "recommended_id_mapping_files": _take_list(llm_rel.get("recommended_id_mapping_files", []), 10),
                },
                "signals": {
                    "candidate_train_files": _take_list(llm_signals.get("candidate_train_files", []), 15),
                    "candidate_test_files": _take_list(llm_signals.get("candidate_test_files", []), 15),
                    "candidate_submission_files": _take_list(llm_signals.get("candidate_submission_files", []), 10),
                    "notes": _take_list(llm_signals.get("notes", []), 20),
                },
            },
            "summaries": summaries,
            "profiles_light": profiles_light,
            "id_format_analysis": id_format_analysis,
        }

        description = {
            "name": description_analysis.get("name"),
            "task": description_analysis.get("task"),
            "output_data": description_analysis.get("output_data"),
        }

        profiling_compact = json.dumps(compact, ensure_ascii=False)
        prompt = self.template.format(
            description=json.dumps(description, ensure_ascii=False),
            profiling_compact=profiling_compact,
        )
        self.manager.save_and_log_states(prompt, "profiling_summarizer_prompt.txt")
        return prompt

    def parse(self, response: str) -> Dict[str, Any]:
        try:
            clean = response.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(clean)
        except Exception:
            parsed = {"error": "Invalid JSON from LLM", "raw_response": response}
        return parsed
