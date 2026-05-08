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
- Prefer reading saved artifacts over recomputing anything.

## AVAILABLE CONTEXT
- Iteration type: {iteration_type}
- Diagnosis round: {round_index}/{max_rounds}
- Output folder (absolute): {output_folder}
- States folder (absolute): {states_dir}
- Artifacts folder (absolute): {artifacts_dir}
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

### Previous diagnosis summary
```json
{previous_diagnosis_json}
```

## REQUIREMENTS
1. Output ONLY executable Python code. No markdown fences.
2. The script must stay lightweight:
   - Allowed: reading files, parsing stdout/stderr, scanning directories, lightweight dataframe/statistics work, loading saved checkpoints for lightweight inference or error analysis on a bounded sample.
   - Avoid: retraining, long inference loops, network calls, downloads, exhaustive dataset sweeps unless already precomputed.
3. The script must inspect whatever evidence exists in the output folder, artifacts folder, and states folder.
4. Prefer the artifact contract from the assembler:
   - `artifacts/metadata.json`
   - saved model/checkpoint paths
   - preprocessing assets
   - validation predictions / labels
   - training history
5. If some evidence is missing, the script must degrade gracefully and emit `unknown` / empty lists instead of guessing.
6. The script should infer the task type if possible.
7. The script must write its final JSON to:
   `{diagnosis_json_path}`
8. The script must print exactly one JSON object between these markers:
   - {diagnosis_start}
   - {diagnosis_end}
9. Before printing the final JSON, the script should emit concise debug logs that explain:
   - which artifacts/files were found and used
   - which metrics or history were recovered
   - any missing evidence that blocks a stronger conclusion
   - any lightweight extra analysis performed (for example confusion matrix, hard examples, segment metrics)
10. The script should decide whether another diagnosis round is needed:
   - Set `stop_diagnosis=true` when the top bottleneck is specific enough to guide the research phase.
   - Set `needs_another_round=true` only when evidence is still insufficient and suggest a narrow `next_round_focus`.
   - Across rounds, focus on filling the biggest evidence gap rather than repeating the same checks.

## OUTPUT CONTRACT
The JSON object must be valid and have this schema:
{{
  "success": bool,
  "task_type": "classification|regression|ranking|segmentation|forecasting|nlp|vision|tabular|unknown",
  "primary_metric": {{"name": str, "value": number|null, "higher_is_better": bool|null}},
  "fit_status": "underfit|overfit|balanced|unknown",
  "generalization_gap": number|null,
  "confidence": "low|medium|high",
  "top_bottleneck": {{"area": str, "reason": str, "evidence": [str]}},
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
  "artifact_paths": {{
    "model": [str],
    "preprocessor": [str],
    "validation_predictions": [str],
    "training_history": [str],
    "metadata": [str],
    "other": [str]
  }},
  "debug_actions_taken": [str],
  "next_round_focus": [str],
  "needs_another_round": bool,
  "stop_diagnosis": bool,
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
- If a prior round already identified a likely bottleneck, only request another round when new evidence could materially increase confidence.
- Print the final JSON only once, after the diagnostic logs.
"""

    def build(
        self,
        *,
        description_analysis: Dict[str, Any],
        profiling_summary: Dict[str, Any],
        baseline_code: str,
        baseline_stdout: str,
        latest_execution_result: Optional[Dict[str, Any]] = None,
        previous_diagnosis: Optional[Dict[str, Any]] = None,
        round_index: int = 1,
        max_rounds: int = 1,
        iteration_type: Optional[str] = None,
        output_folder: str,
        states_dir: str,
        artifacts_dir: str,
        diagnosis_json_path: str,
    ) -> str:
        return self.template.format(
            iteration_type=iteration_type or "default",
            round_index=int(round_index),
            max_rounds=int(max_rounds),
            output_folder=output_folder,
            states_dir=states_dir,
            artifacts_dir=artifacts_dir,
            latest_phase_name=(latest_execution_result or {}).get("phase_name", "unknown"),
            latest_attempt=(latest_execution_result or {}).get("attempt", "unknown"),
            description_json=json.dumps(description_analysis or {}, ensure_ascii=False, indent=2),
            profiling_summary_json=json.dumps(profiling_summary or {}, ensure_ascii=False, indent=2),
            latest_execution_json=json.dumps(latest_execution_result or {}, ensure_ascii=False, indent=2),
            baseline_code=baseline_code or "",
            baseline_stdout=(baseline_stdout or "")[-6000:],
            previous_diagnosis_json=json.dumps(previous_diagnosis or {}, ensure_ascii=False, indent=2),
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
