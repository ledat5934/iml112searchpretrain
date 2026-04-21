# src/iML/prompts/assembler_prompt.py
import json
from pathlib import Path
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
4.  **MANDATORY DEPLOYMENT ARTIFACTS**: In addition to `submission.csv`, the script MUST create a folder named `deployment` at this absolute path: `{deployment_path}` (same level as `submission.csv`).
        - The `deployment` folder MUST contain enough artifacts to run preprocessing + inference later WITHOUT retraining.
        - **Use canonical manifest name exactly**: `deployment/manifest.json` (NOT `deployment_manifest.json` and no truncated JSON).
                - **Manifest contract MUST follow this exact structural style (JSON object; each artifact entry is an object with `filename`)**:
                    ```json
                    {{
                        "model": {{"filename": "model.joblib", "role": "model"}},
                        "preprocessor": {{"filename": "preprocessor.joblib", "role": "preprocessing"}},
                        "label_encoder": {{"filename": "label_encoder.joblib", "role": "preprocessing"}},
                        "metadata": {{"filename": "metadata.json", "role": "metadata"}}
                    }}
                    ```
                - Do NOT use loose path-style manifest keys such as `model_path`, `vectorizer_path`, `model_artifact`, `preprocessing_artifacts`.
        - At minimum, persist:
            - trained model weights/object (e.g., `.pt`, `.pkl`, `.joblib`, etc.)
            - preprocessing artifacts (encoders/scalers/tokenizer/config/feature map) required for consistent preprocessing
            - inference configuration/metadata (feature columns, target mapping, label encoder mapping, model class, versions if available)
            - `manifest.json` listing all artifacts and how to load them.
        - **PREDICT OUTPUT CONTRACT (downstream `deployment.py` MUST follow):**
            - For **classification**: return `[{{"label": <decoded_label>, "confidence": <float in [0,1]>}}, ...]`
            - For **regression**: return `[{{"value": <float>}}, ...]`
            - To make this possible, the trained artifacts MUST support it:
                * Save `metadata.json` with at least:
                    - `"task_type"`: `"classification"` or `"regression"`
                    - For classification: `"class_labels"` (list of decoded labels in model order)
                      and persist any label encoder used during training (e.g. `label_encoder.joblib`).
                * For classification, prefer a model that exposes `predict_proba` (e.g. RandomForest,
                  LogisticRegression, XGBoost, LightGBM, CatBoost, or a NN with softmax head).
                  If using a model without native probabilities (e.g. linear SVM), wrap it with
                  `CalibratedClassifierCV` and persist the calibrated estimator so the deployment
                  module can compute confidence without retraining.
        - The script MUST include a reusable loading path/function that can perform inference from `deployment` artifacts without calling any training routine.
        - Before writing new artifacts, clear stale deployment outputs in `{deployment_path}` (or overwrite deterministically) so old files cannot create false success.
        - After writing artifacts, MUST validate deployment integrity:
            - `manifest.json` is valid JSON (`json.load` succeeds)
            - every artifact referenced in manifest exists on disk
            - at least one model artifact and one preprocessing artifact are loadable
        - The script MUST fail (stderr + non-zero exit) if required deployment artifacts are missing/invalid after save.
5.  **Error Handling (NO SILENT FAILURE)**:
    - Maintain a single `try...except` block for robust execution.
    - If ANY exception occurs, you MUST print the error to stderr and **exit with a non-zero status code** (`sys.exit(1)`).
    - **NEVER** "handle errors" by creating a placeholder/empty `submission.csv` (e.g., using `sample_submission.csv` columns with zero rows).
    - **NEVER** swallow exceptions and continue as if successful.
6.  **Submission Integrity (MUST NOT BE EMPTY)**:
    - You MUST only write `submission.csv` after predictions are successfully produced.
    - After writing, verify `submission.csv` is not empty (has at least 1 data row, not just header).
    - If a `sample_submission.csv` exists in the dataset paths, validate that the produced submission has the same columns/order.
    - If any submission validation fails, treat it as a failure: print an error to stderr and `sys.exit(1)`.
7.  **Clarity**: Ensure the final script is clean and well-structured.
8.  **Sample Submission File**: Sample submission file given is for template reference (Columns) only. You have to use the test data or test file to generate predictions and your right submission file. In some cases, you must browse the test image folder to get the IDs and data.
9.  **Do not add any other code.**
10.  **Data Loading**: Keep the data loading code of the preprocessing code. DO NOT CHANGE THE FILE PATHS FROM THE ORIGINAL CODE.
11. **NO PARTIAL SCRIPT OUTPUT**: Do not output import-only or stub code. Return a complete executable script with training, submission generation, deployment saving, and deployment validation.

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

        deployment_path = str((Path(output_path).parent / "deployment").resolve())

        prompt = self.template.format(
            dataset_name=description.get('name', 'N/A'),
            file_paths=description.get('link to the dataset', []),
            output_data_format=description.get('output_data', 'N/A'),
            original_code=original_code,
            output_path=output_path,
            deployment_path=deployment_path,
            retry_context=retry_context,
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
