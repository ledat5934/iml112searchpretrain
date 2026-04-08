import json
from typing import Any, Dict, Optional

from .base_prompt import BasePrompt


class ResearchMutationPrompt(BasePrompt):
    """
    Prompt to propose a SMALL mutation to a working baseline pipeline script,
    adding timeboxed proxy-eval (soft stop) and printing a machine-readable proxy result.
    """

    PROXY_START = "===PROXY_RESULT_START==="
    PROXY_END = "===PROXY_RESULT_END==="

    def default_template(self) -> str:
        return """You are an ML code research agent.

You are given a WORKING baseline Python pipeline script that trains/evaluates and writes a submission file.

Your task depends on MODE:
- MODE = "instrument_only": DO NOT change modeling logic/hyperparameters. Only add proxy time budget (soft stop) and proxy result printing.
- MODE = "mutate": produce ONE mutated version of the script with exactly ONE SMALL, SAFE change intended to improve validation performance, plus the proxy time budget and proxy result printing.

MODE:
{mode}

## HARD CONSTRAINTS
- Keep the script runnable end-to-end.
- Do NOT change the dataset split logic: use the SAME split as the baseline (same random_state=42 behavior).
- The mutated script MUST keep producing `submission.csv` in the current working directory (same as baseline).
- Add a PROXY TIME BUDGET (soft stop): stop training when time budget is exceeded, but exit cleanly and still compute/report the best metric achieved so far.
- The proxy time budget is: {proxy_time_budget_sec} seconds.
- No network calls. No downloads.

## PROXY RESULT OUTPUT (CRITICAL)
At the end of the run, the script MUST print exactly one JSON object between these markers:
- Print a line: {proxy_start}
- Print the JSON object (valid JSON)
- Print a line: {proxy_end}

The JSON MUST be small and include at least:
{{
  "success": bool,
  "proxy_metric": {{"name": str, "value": number|null, "higher_is_better": bool|null}},
  "fallback_train_loss": number|null,
  "elapsed_sec": number,
  "notes": [str]
}}

Guidance:
- Prefer a validation metric aligned with the task if available (e.g., logloss/AUC/RMSE).
- If you cannot compute validation metric within time budget, set proxy_metric.value=null and provide fallback_train_loss if available.

## MUTATION GUIDANCE (choose ONE)
Pick ONE small change, such as:
- slightly different learning rate / scheduler
- early stopping patience
- regularization (weight decay, dropout, L2)
- label smoothing (classification)
- class weights (if imbalance is evident)
- simple feature normalization/standardization step (tabular)
- light data augmentation tweak (image)
Do NOT do multiple changes.

## BASELINE SCRIPT
```python
{baseline_code}
```

## OUTPUT FORMAT
- Output ONLY the mutated full Python script.
- No markdown fences.
"""

    def build(
        self,
        *,
        baseline_code: str,
        proxy_time_budget_sec: int,
        mode: str = "mutate",
    ) -> str:
        class _SafeDict(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        values = _SafeDict(
            baseline_code=baseline_code or "",
            proxy_time_budget_sec=int(proxy_time_budget_sec),
            mode=str(mode or "mutate"),
            proxy_start=self.PROXY_START,
            proxy_end=self.PROXY_END,
        )
        return self.template.format_map(values)

    def parse(self, response: str) -> str:
        if "```python" in response:
            return response.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in response:
            return response.split("```", 1)[1].split("```", 1)[0].strip()
        return response.strip()

