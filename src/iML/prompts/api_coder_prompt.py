import json
from typing import Any, Dict

from .base_prompt import BasePrompt


class APICoderPrompt(BasePrompt):
    """Prompt for generating a FastAPI wrapper around the deployment bundle."""

    def default_template(self) -> str:
        return """
You are a backend engineer.
Generate a single FastAPI service file named `api.py`.

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

- Schemas code:
```python
{schemas_code}
```

## REQUIREMENTS
- Expose `POST /predict` only.
- Validate input using pydantic models from `schemas.py` when possible.
- Load `AutoPredictor` from local `predictor.py`.
- The route must:
  1. validate input
  2. preprocess and predict through the predictor
  3. return JSON-safe output
- Include a lightweight `GET /health` endpoint.
- If model artifacts are missing, fail loudly with clear errors.
- Return only Python code.
"""

    def build(
        self,
        description_analysis: Dict[str, Any],
        guideline: Dict[str, Any],
        predictor_code: str,
        schemas_code: str,
        iteration_type: str | None = None,
    ) -> str:
        prompt = self.template.format(
            iteration_type=iteration_type or "default",
            description_json=json.dumps(description_analysis or {}, indent=2, ensure_ascii=False),
            guideline_json=json.dumps(guideline or {}, indent=2, ensure_ascii=False),
            predictor_code=predictor_code or "",
            schemas_code=schemas_code or "",
        )
        self.manager.save_and_log_states(prompt, "deployment/api_coder_prompt.txt")
        return prompt

    def parse(self, response: str) -> str:
        if "```python" in response:
            code = response.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in response:
            code = response.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            code = response.strip()
        self.manager.save_and_log_states(code, "deployment/api_generated.py")
        return code + ("\n" if code and not code.endswith("\n") else "")
