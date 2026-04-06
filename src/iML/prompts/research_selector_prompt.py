import json
from typing import Any, Dict

from .base_prompt import BasePrompt


class ResearchSelectorPrompt(BasePrompt):
    """Prompt for selecting the next research experiments from scored proposals."""

    def default_template(self) -> str:
        return """
You are a research selection agent.
Choose which proposals should move into experiment coding first.

## CONTEXT
- Max selected proposals: {top_k}
- Proposals:
```json
{proposals_json}
```

- Proxy scores:
```json
{proxy_scores_json}
```

## OUTPUT
Return valid JSON only:
```json
{{
  "selection_summary": "short paragraph",
  "selected_proposals": [
    {{
      "proposal_id": "exp_001",
      "priority_rank": 1,
      "selection_reason": "why this is selected now"
    }}
  ],
  "rejected_proposals": [
    {{
      "proposal_id": "exp_004",
      "reason": "why it is rejected or deferred"
    }}
  ],
  "primary_proposal_id": "exp_001"
}}
```
"""

    def build(self, proposals: Dict[str, Any], proxy_scores: Dict[str, Any], top_k: int) -> str:
        prompt = self.template.format(
            top_k=top_k,
            proposals_json=json.dumps(proposals or {}, indent=2, ensure_ascii=False),
            proxy_scores_json=json.dumps(proxy_scores or {}, indent=2, ensure_ascii=False),
        )
        self.manager.save_and_log_states(prompt, "research/research_selector_prompt.txt")
        return prompt

    def parse(self, response: str) -> Dict[str, Any]:
        try:
            cleaned = response.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(cleaned)
        except Exception as exc:
            parsed = {"error": f"Invalid JSON response from LLM: {exc}", "raw_response": response}
        try:
            self.manager.save_and_log_states(
                json.dumps(parsed, indent=2, ensure_ascii=False),
                "research/research_selection.json",
            )
        except Exception:
            pass
        return parsed
