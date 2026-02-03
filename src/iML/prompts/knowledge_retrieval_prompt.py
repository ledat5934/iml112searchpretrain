# src/iML/prompts/knowledge_retrieval_prompt.py
import json
from typing import Dict, Any

from .base_prompt import BasePrompt


class KnowledgeRetrievalPrompt(BasePrompt):
    """
    Prompt handler to synthesize iteration-specific knowledge packs.
    """

    def default_template(self) -> str:
        return """You are an expert ML engineer. Produce an ITERATION-SPECIFIC KNOWLEDGE PACK to guide planning.
This knowledge pack should summarize best practices and pitfalls for the given task context and iteration type.

ITERATION TYPE: {iteration_type}

## TASK CONTEXT (JSON)
```json
{task_context_json}
```

## MODEL SHORTLIST (JSON) - optional
```json
{model_suggestions_json}
```

## ARCHITECTURE PATTERNS (JSON) - optional
```json
{architecture_suggestions_json}
```

## OUTPUT REQUIREMENTS
Return a SINGLE JSON object with the following fields (add more if needed, but keep it concise):
{{
  "iteration_type": "{iteration_type}",
  "preprocessing_recommendations": [
    {{"step": "...", "details": "..."}}
  ],
  "training_recommendations": [
    {{"item": "...", "details": "..."}}
  ],
  "validation_and_metrics": {{
    "metrics": ["..."],
    "split_strategy": "...",
    "notes": "..."
  }},
  "input_contract_notes": [
    "..."
  ],
  "submission_notes": [
    "..."
  ],
  "common_pitfalls": [
    "..."
  ],
  "quality_checklist": [
    "..."
  ],
  "sources_or_keywords": [
    "..."
  ],
  "notes": "This pack is guidance only; do not include example code."
}}

IMPORTANT:
- Output MUST be valid JSON (no markdown, no code fences).
- Do NOT include any example code.
"""

    def build(
        self,
        iteration_type: str,
        task_context: Dict[str, Any],
        model_suggestions: Dict[str, Any] | None = None,
        architecture_suggestions: Dict[str, Any] | None = None,
    ) -> str:
        return self.template.format(
            iteration_type=iteration_type or "unknown",
            task_context_json=json.dumps(task_context or {}, indent=2, ensure_ascii=False),
            model_suggestions_json=json.dumps(model_suggestions or {}, indent=2, ensure_ascii=False),
            architecture_suggestions_json=json.dumps(architecture_suggestions or {}, indent=2, ensure_ascii=False),
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
