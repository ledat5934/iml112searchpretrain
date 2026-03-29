# src/iML/prompts/task_schema_prompt.py
import json
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

    def build(
        self,
        description_text: str,
        description_analysis: Dict[str, Any],
        directory_structure: str,
    ) -> str:
        return self.template.format(
            description_text=description_text or "",
            description_analysis_json=json.dumps(description_analysis or {}, indent=2, ensure_ascii=False),
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
