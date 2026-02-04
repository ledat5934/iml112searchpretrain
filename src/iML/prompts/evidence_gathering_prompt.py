# src/iML/prompts/evidence_gathering_prompt.py
import json
from typing import Dict, Any

from .base_prompt import BasePrompt


class EvidenceGatheringPrompt(BasePrompt):
    """
    Prompt handler to generate a lightweight evidence-gathering script.
    """

    def default_template(self) -> str:
        return """You are a debugging evidence agent. Write a SHORT Python script to inspect data/schema related to the error.
The script should be safe, fast, and only print evidence to stdout.

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

## REQUIREMENTS
- Print key evidence only (columns, dtypes, sample rows, file existence).
- Handle missing files gracefully (print a clear message).
- Avoid heavy computation or training.
- Do NOT modify or write dataset files.
- Output ONLY a Python script (no markdown).
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

    def parse(self, response: str) -> str:
        if "```python" in response:
            return response.split("```python")[1].split("```")[0].strip()
        if "```" in response:
            return response.split("```")[1].split("```")[0].strip()
        return response.strip()
