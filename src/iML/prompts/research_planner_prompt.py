import json
from typing import Any, Dict

from .base_prompt import BasePrompt


class ResearchPlannerPrompt(BasePrompt):
    """Prompt for generating structured post-baseline research proposals."""

    def default_template(self) -> str:
        return """
You are a research planner for an AutoML system.
Generate structured post-baseline improvement proposals.

## CONTEXT
- Iteration type: {iteration_type}
- Baseline summary:
```json
{baseline_summary_json}
```

- Data contract:
```json
{data_contract_json}
```

- Guideline:
```json
{guideline_json}
```

- Task context:
```json
{task_context_json}
```

## GOAL
Produce a compact search space for the research phase after a baseline script has already run successfully.

## RULES
- Prefer model-side improvements first.
- Set `needs_preprocessing_change` to true only when you have a strong reason.
- Proposals must be implementable by another coding agent.
- Tailor proxy families:
  - `custom_nn` / `custom_nn_search`: prefer `naswot`, `snip`, `synflow`, structural heuristics
  - `pretrained`: prefer memory, throughput, compatibility, frozen-backbone heuristics
  - `traditional`: prefer meta-feature heuristics, tiny holdout, or low-cost tuning
- Keep the number of proposals between 4 and {max_proposals}.

## OUTPUT FORMAT
Return valid JSON only:
```json
{{
  "research_focus": "one paragraph",
  "guardrails": [
    "short rule 1",
    "short rule 2"
  ],
  "proposals": [
    {{
      "proposal_id": "exp_001",
      "title": "short title",
      "objective": "what to improve",
      "rationale": "why this idea fits the task",
      "changes": [
        "specific modeling change",
        "specific optimization change"
      ],
      "expected_gain": "what metric or behavior may improve",
      "risk_level": "low/medium/high",
      "cost_level": "low/medium/high",
      "needs_preprocessing_change": false,
      "proxy_family": ["naswot", "synflow"],
      "validation_focus": ["metric", "stability"],
      "applicable_iteration": "{iteration_type}"
    }}
  ]
}}
```
"""

    def build(
        self,
        baseline_summary: Dict[str, Any],
        data_contract: Dict[str, Any],
        guideline: Dict[str, Any],
        task_context: Dict[str, Any],
        iteration_type: str | None,
        max_proposals: int,
    ) -> str:
        prompt = self.template.format(
            iteration_type=iteration_type or "default",
            baseline_summary_json=json.dumps(baseline_summary or {}, indent=2, ensure_ascii=False),
            data_contract_json=json.dumps(data_contract or {}, indent=2, ensure_ascii=False),
            guideline_json=json.dumps(guideline or {}, indent=2, ensure_ascii=False),
            task_context_json=json.dumps(task_context or {}, indent=2, ensure_ascii=False),
            max_proposals=max_proposals,
        )
        self.manager.save_and_log_states(prompt, "research/research_planner_prompt.txt")
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
                "research/research_proposals.json",
            )
        except Exception:
            pass
        return parsed
