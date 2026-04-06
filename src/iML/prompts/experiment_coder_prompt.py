import json
from typing import Any, Dict

from .base_prompt import BasePrompt


class ExperimentCoderPrompt(BasePrompt):
    """Prompt for generating a cached-data experiment script from a selected proposal."""

    def default_template(self) -> str:
        return """
You are a machine learning research engineer.
Generate a COMPLETE Python script named `train.py` for a single experiment candidate.

This script lives inside `research_workspace/experiments/<proposal_id>/`.
Sibling directories:
- `../../artifacts/cache/`
- `../../metadata/data_contract.json`
- `../../artifact_io.py`

## GOAL
Create one experiment script that loads cached preprocessing outputs and applies the selected proposal.

## CONTEXT
- Iteration type: {iteration_type}
- Selected proposal:
```json
{proposal_json}
```

- Baseline summary:
```json
{baseline_summary_json}
```

- Data contract:
```json
{data_contract_json}
```

- Baseline modeling code:
```python
{baseline_modeling_code}
```

## HARD REQUIREMENTS
- Import `load_payload` from `artifact_io.py`.
- Load cache from `workspace_root / "artifacts" / "cache"`.
- Keep the script focused on the selected proposal. Do not redesign the whole pipeline.
- Assume cache payload may be dict/list/tuple; write small helper logic to unpack common patterns.
- You MUST persist quantitative metrics for downstream automatic selection:
  - print `PRIMARY_METRIC_NAME: <metric_name>`
  - print `PRIMARY_METRIC_VALUE: <float_value>`
  - write `experiment_metrics.json` inside the experiment directory with at least:
    - `primary_metric_name`
    - `primary_metric_value`
    - `all_metrics` (dict of numeric metrics)
- Save predictions to `submission.csv` inside the experiment directory.
- Wrap main execution in `if __name__ == "__main__":` with `try/except` and `sys.exit(1)` on failure.
- If the cache contract is ambiguous, write defensive validation code and fail loudly.

Return only Python code.
"""

    def build(
        self,
        proposal: Dict[str, Any],
        baseline_summary: Dict[str, Any],
        data_contract: Dict[str, Any],
        baseline_modeling_code: str,
        iteration_type: str | None,
    ) -> str:
        prompt = self.template.format(
            iteration_type=iteration_type or "default",
            proposal_json=json.dumps(proposal or {}, indent=2, ensure_ascii=False),
            baseline_summary_json=json.dumps(baseline_summary or {}, indent=2, ensure_ascii=False),
            data_contract_json=json.dumps(data_contract or {}, indent=2, ensure_ascii=False),
            baseline_modeling_code=baseline_modeling_code or "",
        )
        self.manager.save_and_log_states(prompt, "research/experiment_coder_prompt.txt")
        return prompt

    def parse(self, response: str) -> str:
        if "```python" in response:
            code = response.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in response:
            code = response.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            code = response.strip()
        self.manager.save_and_log_states(code, "research/experiment_candidate.py")
        return code
