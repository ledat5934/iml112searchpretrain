import json
from typing import Any, Dict

from .base_prompt import BasePrompt


class APITestPrompt(BasePrompt):
    """Prompt for generating a deployment API validation script."""

    def default_template(self) -> str:
        return """
You are a QA engineer for ML inference services.
Generate a Python validation script named `validate_api.py`.

## CONTEXT
- Iteration type: {iteration_type}
- Description analysis:
```json
{description_json}
```

- Guideline:
```json
{guideline_json}
```

- Predictor code:
```python
{predictor_code}
```

- API code:
```python
{api_code}
```

## REQUIREMENTS
- The script must import the generated FastAPI app from `api.py`.
- It must use `fastapi.testclient.TestClient`.
- It must read `metadata/sample_request.json`.
- It must call:
  - `GET /health`
  - `POST /predict`
- It must fail with non-zero exit if:
  - import fails
  - routes fail
  - response schema is invalid
  - prediction output is empty
- It must print a short success summary on success.
- Return only Python code.
"""

    def build(
        self,
        description_analysis: Dict[str, Any],
        guideline: Dict[str, Any],
        predictor_code: str,
        api_code: str,
        iteration_type: str | None = None,
    ) -> str:
        prompt = self.template.format(
            iteration_type=iteration_type or "default",
            description_json=json.dumps(description_analysis or {}, indent=2, ensure_ascii=False),
            guideline_json=json.dumps(guideline or {}, indent=2, ensure_ascii=False),
            predictor_code=predictor_code or "",
            api_code=api_code or "",
        )
        self.manager.save_and_log_states(prompt, "deployment/api_test_prompt.txt")
        return prompt

    def parse(self, response: str) -> str:
        if "```python" in response:
            code = response.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in response:
            code = response.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            code = response.strip()
        self.manager.save_and_log_states(code, "deployment/api_validation_generated.py")
        return code + ("\n" if code and not code.endswith("\n") else "")
