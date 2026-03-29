import json
from typing import Any, Dict

from .base_prompt import BasePrompt
from ..utils.utils import smart_truncate_error


class MonolithicCoderPrompt(BasePrompt):
    """Prompt handler to generate a single end-to-end script (preprocessing + modeling)."""

    def default_template(self) -> str:
        return """
You are a senior ML engineer. Generate ONE complete and executable Python script that covers BOTH preprocessing and modeling.
There will be NO subsequent assembly step, so your script must be production-ready on its own.

## DATASET CONTEXT
- Dataset Name: {dataset_name}
- Task Description: {task_desc}
- File Paths: {file_paths}
- Data File Description: {data_file_description}
- Submission File Description: {submission_file_description}

## GUIDANCE (if available)
### Preprocessing Guidance
{preprocessing_guidance}

### Modeling Guidance
{modeling_guidance}

## REQUIREMENTS
1. Include all imports (pandas, numpy, torch/sklearn/etc. as needed).
2. Implement `preprocess_data(file_paths: dict)` that returns the appropriate objects for the iteration type.
3. Implement `train_and_predict(...)` that trains the model and returns predictions ready for submission.
4. Wrap the main execution block in `if __name__ == "__main__":` with try/except and `sys.exit(1)` on failure.
5. Always call `preprocess_data()` inside `main`, pass its outputs to `train_and_predict`, and create `submission.csv` under the current working directory.
6. Respect iteration-specific instructions:
{iteration_notes}
7. Follow ID formatting instructions exactly. If IDs need file extensions, preserve them. Otherwise, strip them.
8. Limit comments. Do NOT create dummy data. Use only the provided paths.
9. Print a concise validation metric before writing the submission.
10. Submission must match the sample submission schema (column order/count).

## OUTPUT FORMAT
Return ONLY the final Python script (no explanations, no markdown).
"""

    def build(
        self,
        guideline: Dict[str, Any],
        description: Dict[str, Any],
        previous_code: str = None,
        error_message: str = None,
        iteration_type: str = None,
    ) -> str:
        guideline = guideline or {}
        preprocessing_guidance = json.dumps(guideline.get("preprocessing", {}), indent=2)
        modeling_section = guideline.get("modeling", {})
        modeling_guidance = json.dumps(modeling_section, indent=2)

        iteration_notes = self._build_iteration_notes(iteration_type, modeling_section)

        prompt = self.template.format(
            dataset_name=description.get("name", "N/A"),
            task_desc=description.get("task", "N/A"),
            file_paths=description.get("link to the dataset", []),
            data_file_description=description.get("data file description", "N/A"),
            submission_file_description=description.get("submission file description", "N/A"),
            preprocessing_guidance=preprocessing_guidance,
            modeling_guidance=modeling_guidance,
            iteration_notes=iteration_notes,
        )

        if previous_code and error_message:
            max_lines = getattr(self.manager.config, "max_error_lines_for_llm", 20)
            max_chars = getattr(self.manager.config, "max_error_message_length", 2048)
            truncated_error = smart_truncate_error(error_message, max_lines=max_lines, max_chars=max_chars)
            prompt += f"""
## PREVIOUS CODE (FAILED)
```python
{previous_code}
```

## ERROR SUMMARY
```
{truncated_error}
```

Generate a corrected complete script:
"""

        # Include full description JSON for additional context.
        try:
            description_json = json.dumps(description, indent=2, ensure_ascii=False)
        except Exception:
            description_json = json.dumps(description, indent=2)
        prompt += f"\n\n## FULL DESCRIPTION ANALYSIS\n```json\n{description_json}\n```\n"

        self.manager.save_and_log_states(prompt, "monolithic_coder_prompt.txt")
        return prompt

    def parse(self, response: str) -> str:
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response.strip()

        self.manager.save_and_log_states(code, "monolithic_coder_response.py")
        return code

    def _build_iteration_notes(self, iteration_type: str, modeling_guidance: Dict[str, Any]) -> str:
        if iteration_type == "traditional":
            return "- Use scikit-learn / LightGBM style pipelines with deterministic splits."
        if iteration_type == "custom_nn":
            return "- Build a PyTorch model from scratch with batching and early stopping."
        if iteration_type == "custom_nn_search":
            return "- Adapt the suggested architecture; ensure tensor dimensions align with retrieved patterns."
        if iteration_type == "pretrained":
            return "- Load HuggingFace/torchvision models with finite DataLoaders (NO IterableDataset)."
        return "- Use reasonable defaults for the detected task."

