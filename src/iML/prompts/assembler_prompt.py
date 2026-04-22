# src/iML/prompts/assembler_prompt.py
import json
from typing import Dict, Any

from .base_prompt import BasePrompt
from ..utils.utils import smart_truncate_error

class AssemblerPrompt(BasePrompt):
    """
    Prompt handler to assemble and fix final code.
    """

    def default_template(self) -> str:
        """Default template to request LLM to rewrite and fix code."""
        return """
You are a senior ML engineer finalizing a project. You have been given a Python script that combines preprocessing and modeling.
Your task is to ensure the script is clean, robust, and correct.

## CONTEXT


## REQUIREMENTS:
1.  **Final Script**: The output must be a single, standalone, executable Python file and it should be run on the real data.
2.  **Validation Score**: If validation data is available, you MUST calculate and print a relevant validation score.
3.  **Absolute Output Path**: The script MUST save `submission.csv` to the following absolute path: `{output_path}`.
4.  **Error Handling (NO SILENT FAILURE)**:
    - Maintain a single `try...except` block for robust execution.
    - If ANY exception occurs, you MUST print the error to stderr and **exit with a non-zero status code** (`sys.exit(1)`).
    - **NEVER** "handle errors" by creating a placeholder/empty `submission.csv` (e.g., using `sample_submission.csv` columns with zero rows).
    - **NEVER** swallow exceptions and continue as if successful.
5.  **Submission Integrity (MUST NOT BE EMPTY)**:
    - You MUST only write `submission.csv` after predictions are successfully produced.
    - After writing, verify `submission.csv` is not empty (has at least 1 data row, not just header).
    - If a `sample_submission.csv` exists in the dataset paths, validate that the produced submission has the same columns/order.
    - If any submission validation fails, treat it as a failure: print an error to stderr and `sys.exit(1)`.
{diagnostics_requirement}
6.  **Clarity**: Ensure the final script is clean and well-structured.
7.  **Sample Submission File**: Sample submission file given is for template reference (Columns) only. You have to use the test data or test file to generate predictions and your right submission file. In some cases, you must browse the test image folder to get the IDs and data.
8.  **Do not add any other code.**
9.  **Data Loading**: Keep the data loading code of the preprocessing code. DO NOT CHANGE THE FILE PATHS FROM THE ORIGINAL CODE.

## DATAFILE STRUCTURE (SUMMARY)
{datafile_structure}



## ORIGINAL CODE:
```python
{original_code}
```
{retry_context}
## INSTRUCTIONS:
Based on the context above, generate the complete and corrected Python code. The output should be ONLY the final Python code.

## FINAL, CORRECTED CODE:
"""

    def build(
        self,
        original_code: str,
        output_path: str,
        description: Dict,
        error_message: str = None,
        iteration_type: str = None,
        datafile_structure: str | None = None,
        prompt_fields: Dict[str, Any] | None = None,
    ) -> str:
        """Build prompt to assemble or fix code."""

        retry_context = ""
        diagnostics_requirement = ""
        if getattr(self.manager, "is_research_phase_enabled", lambda: False)():
            diagnostics_requirement = """6.  **Diagnostics JSON (MUST PRINT)**:
    - Before exiting successfully, print exactly one JSON object between these markers:
      - `===DIAGNOSIS_SUMMARY_START===`
      - `===DIAGNOSIS_SUMMARY_END===`
    - The JSON must be valid and small, and include at least:
      - `success`: bool
      - `validation_metric`: object with `name`, `value`, `higher_is_better` (use nulls if unavailable)
      - `train_metric`: object with `name`, `value`, `higher_is_better` (use nulls if unavailable)
      - `generalization_gap`: number or null
      - `fit_status`: one of `underfit`, `overfit`, `balanced`, `unknown`
      - `suspected_bottlenecks`: short list of strings
      - `notes`: short list of strings
"""
        if error_message:
            # Use smart truncation for error message to save tokens and focus on relevant parts
            max_lines = getattr(self.manager.config, 'max_error_lines_for_llm', 20)
            max_chars = getattr(self.manager.config, 'max_error_message_length', 2048)
            truncated_error = smart_truncate_error(error_message, max_lines=max_lines, max_chars=max_chars)
            # Include dataset paths from description analyzer for context
            dataset_paths = description.get('link to the dataset', [])
            dataset_paths_json = json.dumps(dataset_paths)

            retry_context = f"""
## PREVIOUS ATTEMPT FAILED:
The code above failed with the following error.

### Error Message:
```
{truncated_error}
```

### Dataset Paths (from description analyzer):
{dataset_paths_json}

### FIX INSTRUCTIONS:
1.  Analyze the error message and the original code carefully.
2.  Fix the specific issue that caused the error.
3.  Generate a new, complete, and corrected version of the Python code that resolves the issue and meets all requirements.
4.  If the error indicates a missing module (ModuleNotFoundError/ImportError), modify the final script to wrap critical imports in try/except and, in the except, call subprocess to install the missing package (e.g., `[sys.executable, '-m', 'pip', 'install', '<package>']` with `check=True`), then attempt the import again before proceeding.
"""

        # Add iteration-specific assembly guidance
        prompt_fields = prompt_fields or {}
        iteration_guidance = prompt_fields.get("iteration_guidance") or self._get_iteration_guidance(iteration_type)
        if iteration_guidance:
            additional_context = f"\n\n## ITERATION-SPECIFIC CONTEXT:\n{iteration_guidance}"
            retry_context += additional_context

        prompt = self.template.format(
            dataset_name=description.get('name', 'N/A'),
            file_paths=description.get('link to the dataset', []),
            output_data_format=description.get('output_data', 'N/A'),
            original_code=original_code,
            output_path=output_path,
            retry_context=retry_context,
            diagnostics_requirement=diagnostics_requirement,
            datafile_structure=datafile_structure or "N/A",
        )
        
        # Append full description analysis as JSON context
        try:
            description_json = json.dumps(description, indent=2, ensure_ascii=False)
        except Exception:
            description_json = json.dumps(description, indent=2)
        prompt += f"\n\n## FULL DESCRIPTION ANALYSIS (JSON)\n```json\n{description_json}\n```\n"

        self.manager.save_and_log_states(prompt, "assemble/assembler_prompt.txt")
        return prompt
    
    def _get_iteration_guidance(self, iteration_type: str = None) -> str:
        """Get iteration-specific assembly guidance."""
        if iteration_type == "traditional":
            return """
This iteration focuses on Traditional ML algorithms (XGBoost, LightGBM, CatBoost):
- Ensure proper handling of categorical variables for tree-based models
"""
        elif iteration_type == "custom_nn":
            return """
This iteration focuses on Custom Neural Networks:
- Verify proper neural network architecture implementation
- Ensure training loops with validation monitoring are correctly set up
- Check that loss functions and optimizers are appropriate
- Validate proper batch processing and data loading
- Ensure model checkpointing and early stopping are implemented
"""
        elif iteration_type == "pretrained":
            return """
This iteration focuses on Pretrained Models:
- Verify transfer learning implementation with correct layer freezing
- Check that fine-tuning is properly configured
- Validate model-specific preprocessing is maintained
- Ensure proper adaptation for the target task
"""
        else:
            return ""

    def parse(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response
        
        self.manager.save_and_log_states(code, "final_assembled_code.py")
        return code
