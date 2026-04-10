import json
import re
from typing import Any, Dict

from .base_prompt import BasePrompt


class DeploymentRefactorPrompt(BasePrompt):
    """Prompt for refactoring assembled competition code into a deployment bundle."""

    def default_template(self) -> str:
        return """
You are a senior ML platform engineer.
Refactor the generated training pipeline into a deployment-ready predictor bundle.

## GOAL
Generate a deployment bundle with:
1. `predictor.py`
2. `build_bundle.py`
3. `schemas.py`

## REQUIREMENTS
- `predictor.py` must expose a class `AutoPredictor` with:
  - `load(model_dir)`
  - `predict(records)`
  - `predict_one(record)`
  - `predict_proba(records)` when classification is supported
- `build_bundle.py` must build real artifacts under `artifacts/` by reusing the generated ML logic as much as possible.
- If trained artifacts already exist in the provided artifact source directory, prefer reusing and packaging those artifacts instead of retraining.
- Retraining is only acceptable as a fallback when the provided artifact manifest or files are clearly insufficient for inference.
- `build_bundle.py` must also write:
  - `metadata/inference_contract.json`
  - `metadata/bundle_metadata.json`
  - `metadata/sample_request.json`
- `schemas.py` should define pydantic request/response models used by the API layer.
- The bundle must be designed for inference on new records, not for writing submission.csv.
- Keep the code pragmatic and self-contained.
- Prefer using standard libraries plus pandas/numpy/sklearn/torch if already implied by the generated code.
- Do not create fake model outputs or placeholder predictions.

## OUTPUT FORMAT
Return exactly this multi-file format:
===FILE: predictor.py===
<python code>
===END FILE===
===FILE: build_bundle.py===
<python code>
===END FILE===
===FILE: schemas.py===
<python code>
===END FILE===

Do not wrap the response in markdown fences.
Do not add commentary before, between, or after the file blocks.

## CONTEXT
- Iteration type: {iteration_type}
- Description analysis:
```json
{description_json}
```

- Task schema:
```json
{task_schema_json}
```

- Existing trained artifact manifest:
```json
{artifact_manifest_json}
```

- Existing trained artifact inventory:
```json
{artifact_inventory_json}
```

- Artifact source directory available to the generated code: `{artifact_source_dir}`

- Final assembled code:
```python
{assembled_code}
```
"""

    def build(
        self,
        description_analysis: Dict[str, Any],
        task_schema: Dict[str, Any],
        artifact_manifest: Dict[str, Any],
        artifact_inventory: Dict[str, Any],
        artifact_source_dir: str,
        assembled_code: str,
        iteration_type: str | None = None,
    ) -> str:
        prompt = self.template.format(
            iteration_type=iteration_type or "default",
            description_json=json.dumps(description_analysis or {}, indent=2, ensure_ascii=False),
            task_schema_json=json.dumps(task_schema or {}, indent=2, ensure_ascii=False),
            artifact_manifest_json=json.dumps(artifact_manifest or {}, indent=2, ensure_ascii=False),
            artifact_inventory_json=json.dumps(artifact_inventory or {}, indent=2, ensure_ascii=False),
            artifact_source_dir=artifact_source_dir or "",
            assembled_code=assembled_code or "",
        )
        self.manager.save_and_log_states(prompt, "deployment/deployment_refactor_prompt.txt")
        return prompt

    def parse(self, response: str) -> Dict[str, str]:
        response = (response or "").strip()
        if response.startswith("```") and response.endswith("```"):
            response = re.sub(r"^```[^\n]*\n?", "", response)
            response = re.sub(r"\n?```$", "", response)

        pattern = re.compile(
            r"===FILE:\s*(?P<name>[^=]+?)===\s*(?P<content>.*?)===END FILE===",
            flags=re.DOTALL,
        )
        files: Dict[str, str] = {}
        for match in pattern.finditer(response):
            name = match.group("name").strip()
            content = match.group("content").strip()
            content = re.sub(r"^```(?:python)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            content = content.strip()
            files[name] = content + ("\n" if content and not content.endswith("\n") else "")
        if not files:
            raise ValueError("Could not parse deployment bundle files from LLM response.")
        self.manager.save_and_log_states(
            json.dumps({"files": list(files.keys())}, indent=2, ensure_ascii=False),
            "deployment/deployment_refactor_manifest.json",
        )
        return files
