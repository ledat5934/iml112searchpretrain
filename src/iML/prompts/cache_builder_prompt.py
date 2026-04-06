import json
from typing import Any, Dict

from .base_prompt import BasePrompt


class CacheBuilderPrompt(BasePrompt):
    """Prompt for generating a cache builder script from baseline preprocessing code."""

    def default_template(self) -> str:
        return """
You are a machine learning systems engineer.
Generate a COMPLETE Python script named `cache_builder.py`.

This script lives inside a research workspace next to:
- `baseline_preprocessing.py`
- `artifact_io.py`
- `metadata/data_contract.json`

Your job is to:
1. import `preprocess_data` from `baseline_preprocessing.py`
2. call `preprocess_data(file_paths)`
3. save the returned payload into `artifacts/cache/` using `save_payload` from `artifact_io.py`
4. save metadata files into `metadata/`
5. print a short success message with the cache directory path

## CONTEXT
- Iteration type: {iteration_type}
- Dataset description analysis:
```json
{description_json}
```

- Data contract:
```json
{data_contract_json}
```

- Baseline preprocessing code:
```python
{preprocessing_code}
```

## HARD REQUIREMENTS
- The script must be self-contained and executable.
- Use only real dataset paths from `description_analysis["link to the dataset"]`.
- Do not generate any fake data.
- Use `Path(__file__).resolve().parent` as the workspace root.
- Write cache under `workspace_root / "artifacts" / "cache"`.
- Write metadata under `workspace_root / "metadata"`.
- Normalize the cache contract when possible. Prefer writing a dict payload with keys like:
  - `X_train`, `y_train`
  - `X_valid`, `y_valid`
  - `X_test`
  - `test_ids`
- If the preprocessing output does not already expose those exact keys, build the closest consistent mapping you can and record it in `metadata/cache_summary.json`.
- Save at least:
  - `metadata/cache_summary.json`
  - `metadata/preprocess_runtime_metadata.json`
- `metadata/cache_summary.json` must explicitly list:
  - top-level payload type
  - detected split keys
  - detected target keys
  - whether a validation split is present
- If `preprocess_data` returns a dict/list/tuple/numpy/pandas payload, persist the normalized payload through `save_payload(...)`.
- Wrap main execution in `if __name__ == "__main__":` with `try/except` and `sys.exit(1)` on failure.

Return only Python code.
"""

    def build(
        self,
        description_analysis: Dict[str, Any],
        data_contract: Dict[str, Any],
        preprocessing_code: str,
        iteration_type: str | None = None,
    ) -> str:
        prompt = self.template.format(
            iteration_type=iteration_type or "default",
            description_json=json.dumps(description_analysis or {}, indent=2, ensure_ascii=False),
            data_contract_json=json.dumps(data_contract or {}, indent=2, ensure_ascii=False),
            preprocessing_code=preprocessing_code or "",
        )
        self.manager.save_and_log_states(prompt, "research/cache_builder_prompt.txt")
        return prompt

    def parse(self, response: str) -> str:
        if "```python" in response:
            code = response.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in response:
            code = response.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            code = response.strip()
        self.manager.save_and_log_states(code, "research/cache_builder_generated.py")
        return code
