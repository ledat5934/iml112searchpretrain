# src/iML/prompts/error_triage_prompt.py
import json
from typing import Dict, Any

from .base_prompt import BasePrompt


class ErrorTriagePrompt(BasePrompt):
    """
    Prompt handler to decide whether to gather evidence or debug directly.
    """

    def default_template(self) -> str:
        return """You are a debugging triage agent. Decide whether to gather data/schema evidence or debug directly.

## STDERR
{stderr}

## CODE SNIPPET (FAILED)
```python
{code_snippet}
```

## DATAFILE STRUCTURE (SUMMARY)
{datafile_structure}

## TASK SCHEMA (JSON)
```json
{task_schema_json}
```

## OUTPUT FORMAT (JSON ONLY)
{{
  "action": "direct_debug|gather_evidence",
  "rationale": "...",
  "evidence_focus": ["columns", "paths", "dtypes", "labels", "ids", "time", "schema", "other"]
}}

IMPORTANT:
- Output valid JSON only (no markdown).
- Choose "gather_evidence" for errors likely caused by data/schema (missing columns, wrong dtypes, missing files).
"""

    def build(
        self,
        stderr: str,
        code_snippet: str,
        datafile_structure: str,
        task_schema: Dict[str, Any],
    ) -> str:
        return self.template.format(
            stderr=stderr or "",
            code_snippet=code_snippet or "",
            datafile_structure=datafile_structure or "N/A",
            task_schema_json=json.dumps(task_schema or {}, indent=2, ensure_ascii=False),
        )

    def parse(self, response: str) -> Dict[str, Any]:
        try:
            cleaned = response.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned)
        except Exception as e:
            return {"error": f"Invalid JSON response from LLM: {e}", "raw_response": response}
