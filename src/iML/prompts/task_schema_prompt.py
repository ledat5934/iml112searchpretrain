# src/iML/prompts/task_schema_prompt.py
import json
import re
from typing import Dict, Any

from .base_prompt import BasePrompt


class TaskSchemaPrompt(BasePrompt):
    """
    Prompt handler to infer a task schema from description + profiling evidence.
    """

    def default_template(self) -> str:
        return """You are a senior ML analyst. Your task is to infer a precise TASK SCHEMA from the provided evidence.
You must rely on the description and profiling evidence. Use your best judgment, but be explicit about evidence and uncertainty.

## RAW DESCRIPTION
{description_text}

## DESCRIPTION ANALYSIS (JSON)
```json
{description_analysis_json}
```

## PROFILING SUMMARY (JSON)
```json
{profiling_summary_json}
```

## DIRECTORY STRUCTURE
{directory_structure}

## OUTPUT REQUIREMENTS
Return a SINGLE JSON object with the following fields (add more if needed, but keep it concise):
{{
  "task_overview": {{
    "objective": "...",
    "modality": "tabular|image|text|audio|video|multimodal|unknown",
    "prediction_level": "sample|token|pixel|instance|sequence|unknown"
  }},
  "label_schema": {{
    "format": "column_class|yolo_txt|mask_png|jsonl_spans|sequence_target|unknown",
    "target_columns_or_files": ["..."],
    "notes": "..."
  }},
  "prediction_schema": {{
    "type": "class_label|probabilities|boxes|masks|sequence|unknown",
    "notes": "..."
  }},
  "preferred_input_contract": "data_yaml|dataloader|dataframe|files|unknown",
  "evaluation_spec": {{
    "metrics": ["..."],
    "validation_strategy": "random_split|time_split|group_split|rolling_origin|unknown",
    "leakage_warnings": ["..."]
  }},
  "submission_spec": {{
    "format": "csv|json|coco_json|unknown",
    "id_rules": "with_extensions|without_extensions|unknown",
    "required_columns": ["..."],
    "notes": "..."
  }},
  "constraints": {{
    "random_state_required": true,
    "memory_notes": "...",
    "time_budget_notes": "..."
  }},
  "assumptions": ["..."],
  "evidence": [
    {{"claim": "...", "evidence": "...", "confidence": 0.0}}
  ]
}}

IMPORTANT:
- Output MUST be valid JSON (no markdown, no code fences).
- Do NOT include any example code.
- If uncertain, set fields to "unknown" and add an assumption.
"""

    def _compact_name_list(self, items: list[str], max_keep: int = 18) -> list[str]:
        if not isinstance(items, list):
            return []
        names = [str(x) for x in items if x is not None]
        if len(names) <= max_keep:
            return names

        compact: list[str] = []
        groups: dict[str, list[tuple[int, str]]] = {}
        leftovers: list[str] = []
        for name in names:
            m = re.fullmatch(r"([A-Za-z_]+)(\d+)", name)
            if not m:
                leftovers.append(name)
                continue
            groups.setdefault(m.group(1), []).append((int(m.group(2)), name))

        for _, vals in groups.items():
            vals.sort(key=lambda x: x[0])
            originals = [orig for _, orig in vals]
            if len(originals) >= 6:
                compact.extend([originals[0], originals[1], "...", originals[-2], originals[-1]])
            else:
                compact.extend(originals)

        compact.extend(leftovers)
        if len(compact) > max_keep:
            compact = compact[: max_keep - 1] + ["..."]
        return compact

    def _compact_profiling_summary(self, profiling_summary: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(profiling_summary, dict):
            return profiling_summary
        compact = json.loads(json.dumps(profiling_summary))
        for item in compact.get("key_files", []) or []:
            if isinstance(item, dict) and "columns" in item:
                item["columns"] = self._compact_name_list(item.get("columns") or [])
        return compact

    def build(
        self,
        description_text: str,
        description_analysis: Dict[str, Any],
        profiling_summary: Dict[str, Any],
        directory_structure: str,
    ) -> str:
        profiling_summary_compact = self._compact_profiling_summary(profiling_summary or {})
        return self.template.format(
            description_text=description_text or "",
            description_analysis_json=json.dumps(description_analysis or {}, indent=2, ensure_ascii=False),
            profiling_summary_json=json.dumps(profiling_summary_compact, indent=2, ensure_ascii=False),
            directory_structure=directory_structure or "",
        )

    def parse(self, response: str) -> Dict[str, Any]:
        try:
            cleaned = response.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned)
        except Exception as e:
            return {
                "error": f"Invalid JSON response from LLM: {e}",
                "raw_response": response,
            }
