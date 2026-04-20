import json
from typing import Any, Dict

from .base_prompt import BasePrompt


class ResearchProposalPrompt(BasePrompt):
    """Prompt for proposing concrete post-assembly improvement ideas."""

    def default_template(self) -> str:
        return """You are a machine learning research planner.

The baseline script has already run successfully.
Propose exactly {n_candidates} different improvement directions to try next.

## CONTEXT
- Iteration type: {iteration_type}
- Description analysis:
```json
{description_json}
```

- Profiling summary:
```json
{profiling_summary_json}
```

- Baseline stdout excerpt:
```text
{stdout_excerpt}
```

- Baseline code:
```python
{baseline_code}
```

- Previously attempted research directions:
```json
{previous_directions_json}
```

- Low-cost ablation study summary:
```json
{ablation_summary_json}
```

## RULES
- Proposals must be materially different from one another.
- Proposals must also be materially different from previously attempted directions.
- Use the ablation study to decide which subsystem is most promising to improve first.
- Each proposal should be implementable as a single improved version of the existing script.
- Prefer changes that can be evaluated by a short proxy run.
- Do not propose full architecture rewrites unless clearly justified by the task.
- Keep preprocessing changes minimal unless the profile/stdout strongly suggests preprocessing is the bottleneck.

## OUTPUT FORMAT
Return valid JSON only:
{{
  "research_focus": "short paragraph",
  "proposals": [
    {{
      "proposal_id": "candidate_1",
      "title": "short title",
      "objective": "what to improve",
      "rationale": "why this is promising for this task",
      "changes": ["specific change 1", "specific change 2"],
      "expected_metric": "metric name",
      "risk_level": "low/medium/high"
    }}
  ]
}}
"""

    def build(
        self,
        *,
        description_analysis: Dict[str, Any],
        profiling_summary: Dict[str, Any],
        baseline_code: str,
        stdout_excerpt: str,
        n_candidates: int,
        iteration_type: str | None = None,
        previous_directions: list[Dict[str, Any]] | None = None,
        ablation_summary: Dict[str, Any] | None = None,
    ) -> str:
        return self.template.format(
            iteration_type=iteration_type or "default",
            description_json=json.dumps(description_analysis or {}, indent=2, ensure_ascii=False),
            profiling_summary_json=json.dumps(profiling_summary or {}, indent=2, ensure_ascii=False),
            baseline_code=baseline_code or "",
            stdout_excerpt=(stdout_excerpt or "")[-4000:],
            previous_directions_json=json.dumps(previous_directions or [], indent=2, ensure_ascii=False),
            ablation_summary_json=json.dumps(ablation_summary or {}, indent=2, ensure_ascii=False),
            n_candidates=int(n_candidates),
        )

    def parse(self, response: str) -> Dict[str, Any]:
        cleaned = (response or "").strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
        return json.loads(cleaned)
