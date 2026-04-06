import json
from typing import Any, Dict

from .base_prompt import BasePrompt


class ProxyEstimatorPrompt(BasePrompt):
    """Prompt for ranking research proposals with zero-cost or cheap proxy reasoning."""

    def default_template(self) -> str:
        return """
You are a proxy-estimation analyst for post-baseline AutoML research.
Rank improvement proposals before expensive training.

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

- Research proposals:
```json
{proposals_json}
```

## TASK
Estimate which proposals should be promoted first.

## SCORING RULES
- Use zero-cost proxy thinking when the iteration is custom neural network oriented.
- Use compatibility / memory / throughput / implementation-risk heuristics when the proposal is pretrained-oriented.
- Penalize proposals that need preprocessing changes unless the baseline summary suggests a preprocessing bottleneck.
- Be explicit about whether a proposal is worth promoting before full Kaggle experiments.

## OUTPUT
Return valid JSON only:
```json
{{
  "ranking_rationale": "short paragraph",
  "scored_proposals": [
    {{
      "proposal_id": "exp_001",
      "recommended_action": "promote/reject/defer",
      "proxy_methods": ["naswot", "synflow"],
      "feasibility_score": 0.0,
      "expected_gain_score": 0.0,
      "implementation_risk_score": 0.0,
      "memory_risk_score": 0.0,
      "overall_score": 0.0,
      "notes": [
        "short note 1",
        "short note 2"
      ]
    }}
  ]
}}
```
Use scores in the range 0.0 to 10.0.
"""

    def build(
        self,
        baseline_summary: Dict[str, Any],
        data_contract: Dict[str, Any],
        proposals: Dict[str, Any],
        iteration_type: str | None,
    ) -> str:
        prompt = self.template.format(
            iteration_type=iteration_type or "default",
            baseline_summary_json=json.dumps(baseline_summary or {}, indent=2, ensure_ascii=False),
            data_contract_json=json.dumps(data_contract or {}, indent=2, ensure_ascii=False),
            proposals_json=json.dumps(proposals or {}, indent=2, ensure_ascii=False),
        )
        self.manager.save_and_log_states(prompt, "research/proxy_estimator_prompt.txt")
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
                "research/proxy_scores.json",
            )
        except Exception:
            pass
        return parsed
