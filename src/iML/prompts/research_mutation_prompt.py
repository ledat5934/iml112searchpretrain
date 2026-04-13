import json
from typing import Any, Dict, Optional

from .base_prompt import BasePrompt


class ResearchMutationPrompt(BasePrompt):
    """
    Prompt to propose a SMALL mutation to a working baseline pipeline script,
    adding timeboxed proxy-eval (soft stop) and printing a machine-readable proxy result,
    or converting a ranked proxy candidate into a full-run script.
    """

    PROXY_START = "===PROXY_RESULT_START==="
    PROXY_END = "===PROXY_RESULT_END==="

    def default_template(self) -> str:
        return """You are an ML code research agent.

You are given a WORKING baseline Python pipeline script that trains/evaluates and writes a submission file.

Your task depends on MODE:
- MODE = "instrument_only": DO NOT change modeling logic/hyperparameters. Only add proxy time budget (soft stop) and proxy result printing.
- MODE = "mutate_from_proposal": produce ONE improved version of the script that implements the provided proposal, plus the proxy time budget and proxy result printing.
- MODE = "prepare_full_run": take the selected proxy-ranked candidate script and convert it into a full training/inference script. Keep the chosen improvement, remove proxy-only training limits, and preserve real `submission.csv` generation.

MODE:
{mode}

## HARD CONSTRAINTS
- Keep the script runnable end-to-end.
- Do NOT change the dataset split logic: use the SAME split as the baseline (same random_state=42 behavior).
- In full mode, the improved script MUST keep producing `submission.csv` in the current working directory (same as baseline).
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

## PROPOSAL TO IMPLEMENT
```json
{proposal_json}
```

## ADDITIONAL CONTEXT
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

## TRAINING-TIME BUDGET RULE
- In proxy mode, the 5-minute budget applies to TRAINING ONLY.
- Data loading / preprocessing / feature preparation are NOT part of the 5-minute training budget.
- Start the budget immediately before model.fit()/training loop and stop it immediately after training ends.
- In proxy mode, avoid writing/overwriting the final submission.csv.
- In full mode, keep the real submission.csv behavior.
- In full mode, remove proxy-only early termination and do NOT enforce the proxy training budget inside the script.
- In full mode, remove proxy-result marker printing unless it is also useful for normal logging.

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
        mode: str = "mutate_from_proposal",
        proposal: Optional[Dict[str, Any]] = None,
        description_analysis: Optional[Dict[str, Any]] = None,
        profiling_summary: Optional[Dict[str, Any]] = None,
        stdout_excerpt: str = "",
        iteration_type: Optional[str] = None,
    ) -> str:
        class _SafeDict(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        values = _SafeDict(
            baseline_code=baseline_code or "",
            proxy_time_budget_sec=int(proxy_time_budget_sec),
            mode=str(mode or "mutate_from_proposal"),
            proxy_start=self.PROXY_START,
            proxy_end=self.PROXY_END,
            proposal_json=json.dumps(proposal or {}, ensure_ascii=False, indent=2),
            description_json=json.dumps(description_analysis or {}, ensure_ascii=False, indent=2),
            profiling_summary_json=json.dumps(profiling_summary or {}, ensure_ascii=False, indent=2),
            stdout_excerpt=(stdout_excerpt or "")[-4000:],
            iteration_type=str(iteration_type or "default"),
        )
        return self.template.format_map(values)

    def build_full_run(
        self,
        *,
        candidate_code: str,
        proxy_time_budget_sec: int,
        proposal: Optional[Dict[str, Any]] = None,
        description_analysis: Optional[Dict[str, Any]] = None,
        profiling_summary: Optional[Dict[str, Any]] = None,
        stdout_excerpt: str = "",
        iteration_type: Optional[str] = None,
    ) -> str:
        return self.build(
            baseline_code=candidate_code,
            proxy_time_budget_sec=proxy_time_budget_sec,
            mode="prepare_full_run",
            proposal=proposal,
            description_analysis=description_analysis,
            profiling_summary=profiling_summary,
            stdout_excerpt=stdout_excerpt,
            iteration_type=iteration_type,
        )

    def parse(self, response: str) -> str:
        if "```python" in response:
            return response.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in response:
            return response.split("```", 1)[1].split("```", 1)[0].strip()
        return response.strip()

