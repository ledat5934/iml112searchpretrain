import json
from typing import Any, Dict, Optional

from .base_prompt import BasePrompt


class DiagnosisPhasePrompt(BasePrompt):
    """Prompt for generating a task-adaptive diagnosis script with a fixed output contract."""

    DIAGNOSIS_START = "===DIAGNOSIS_RESULT_START==="
    DIAGNOSIS_END = "===DIAGNOSIS_RESULT_END==="

    def default_template(self) -> str:
        return """You are an ML diagnosis engineer.

Your task is to generate ONE Python script that diagnoses the currently assembled ML solution.

The script must adapt to the problem type, but it MUST keep a fixed JSON output contract.

## GOAL
- Inspect the current solution code, execution logs, available output files, and any lightweight artifacts.
- Diagnose where the current solution likely underperforms.
- Recommend improvement directions at the level of change types, not code patches.
- Do NOT retrain models or run expensive experimentation.
- Do NOT modify the existing assembled solution.

## AVAILABLE CONTEXT
- Iteration type: {iteration_type}
- Output folder (absolute): {output_folder}
- States folder (absolute): {states_dir}
- Latest phase name: {latest_phase_name}
- Latest attempt: {latest_attempt}

### Description analysis
```json
{description_json}
```

### Profiling summary
```json
{profiling_summary_json}
```

### Latest execution result
```json
{latest_execution_json}
```

### Current assembled code
```python
{baseline_code}
```

### Latest stdout excerpt
```text
{baseline_stdout}
```

## REQUIREMENTS
1. Output ONLY executable Python code. No markdown fences.
2. The script must stay lightweight:
   - Allowed: reading files, parsing stdout/stderr, scanning directories, lightweight dataframe/statistics work.
   - Avoid: retraining, long inference loops, network calls, downloads.
3. The script must inspect whatever evidence exists in the output folder and states folder.
4. If some evidence is missing, the script must degrade gracefully and emit `unknown` / empty lists instead of guessing.
5. The script should infer the task type if possible.
6. The script must write its final JSON to:
   `{diagnosis_json_path}`
7. The script must print exactly one JSON object between these markers:
   - {diagnosis_start}
   - {diagnosis_end}

## OUTPUT CONTRACT
The JSON object must be valid and have this schema:
{{
  "success": bool,
  "task_type": "classification|regression|ranking|segmentation|forecasting|nlp|vision|tabular|unknown",
  "primary_metric": {{"name": str, "value": number|null, "higher_is_better": bool|null}},
  "fit_status": "underfit|overfit|balanced|unknown",
  "generalization_gap": number|null,
  "suspected_bottlenecks": [str],
  "underperforming_segments": [
    {{
      "segment": str,
      "metric": str,
      "value": number|null,
      "support": int|null,
      "reason": str
    }}
  ],
  "error_patterns": [str],
  "recommended_change_types": [
    "features" | "sampling" | "augmentation" | "loss" | "optimizer_schedule" |
    "architecture" | "regularization" | "thresholding" | "postprocessing" |
    "validation_strategy" | "data_cleaning" | "preprocessing" | "unknown"
  ],
  "notes": [str],
  "evidence_refs": [str]
}}

## DIAGNOSIS CRITERIA
The script should try to cover these when evidence permits:
- task type and metric direction
- train vs validation signal and generalization gap
- likely underfit/overfit/balanced status
- segment-level weakness by class, range bucket, modality/property, or any detectable split
- obvious bottlenecks from code/logs such as class imbalance, poor regularization, weak architecture fit, threshold issues, preprocessing mismatch, objective mismatch, validation weakness
- improvement directions expressed only as change types

## EVIDENCE POLICY
- Prefer concrete evidence from files/logs over speculation.
- If evidence is insufficient, say so in `notes` and use `unknown`.
- Never claim certainty without evidence.
"""

    def build(
        self,
        *,
        description_analysis: Dict[str, Any],
        profiling_summary: Dict[str, Any],
        baseline_code: str,
        baseline_stdout: str,
        latest_execution_result: Optional[Dict[str, Any]] = None,
        iteration_type: Optional[str] = None,
        output_folder: str,
        states_dir: str,
        diagnosis_json_path: str,
    ) -> str:
        return self.template.format(
            iteration_type=iteration_type or "default",
            output_folder=output_folder,
            states_dir=states_dir,
            latest_phase_name=(latest_execution_result or {}).get("phase_name", "unknown"),
            latest_attempt=(latest_execution_result or {}).get("attempt", "unknown"),
            description_json=json.dumps(description_analysis or {}, ensure_ascii=False, indent=2),
            profiling_summary_json=json.dumps(profiling_summary or {}, ensure_ascii=False, indent=2),
            latest_execution_json=json.dumps(latest_execution_result or {}, ensure_ascii=False, indent=2),
            baseline_code=baseline_code or "",
            baseline_stdout=(baseline_stdout or "")[-6000:],
            diagnosis_json_path=diagnosis_json_path,
            diagnosis_start=self.DIAGNOSIS_START,
            diagnosis_end=self.DIAGNOSIS_END,
        )

    def parse(self, response: str) -> str:
        cleaned = (response or "").strip()
        if "```python" in cleaned:
            return cleaned.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in cleaned:
            return cleaned.split("```", 1)[1].split("```", 1)[0].strip()
        return cleaned
