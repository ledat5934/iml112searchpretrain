# src/iML/prompts/preprocessing_coder_prompt.py
import json
from typing import Dict, Any

from .base_prompt import BasePrompt
from ..utils.utils import smart_truncate_error

class PreprocessingCoderPrompt(BasePrompt):
    """
    Prompt handler to generate Python code for data preprocessing.
    """

    def default_template(self) -> str:
        """Default template to request LLM to generate code."""
        return """
You are a professional Machine Learning Engineer.
Generate complete and executable Python preprocessing code for the dataset below.
{batch_processing_instruction}
IMPORTANT: DO NOT CREATE DUMMY DATA.
## DATASET INFO:
- Name: {dataset_name}
- Task: {task_desc}
- Input: {input_desc}
- Output: {output_desc}
- Data files: {data_file_desc}
- File paths: {file_paths} (LOAD DATA FROM THESE PATHS)

## DATAFILE STRUCTURE (SUMMARY)
{datafile_structure}

## PREPROCESSING GUIDELINES:
{preprocessing_guideline}

{input_contract_notes_section}

{model_input_data_section}

## TARGET INFO:
{target_info}

## REQUIREMENTS:
1. Generate COMPLETE, EXECUTABLE Python code.
2. Include all necessary imports (pandas, scikit-learn, numpy, etc.).
3. Handle file loading exactly as the provided paths, DO NOT CREATE DUMMY DATA FILES.
4. Follow the preprocessing guidelines exactly.
{data_return_format}
6. Include basic error handling and data validation within the function.
7. Limit comments in the code.
8. Preprocess both the train and test data consistently.
9. IMPORTANT: The main execution block (`if __name__ == "__main__":`) should test the function with the actual file paths.
10. **Critical Error Handling**: The main execution block MUST be wrapped in a `try...except` block. If ANY exception occurs, the script MUST print the error and then **exit with a non-zero status code** using `sys.exit(1)`.
11. DO NOT USE NLTK
12. Sample submission file given is for template reference (Columns) only. You have to use the test data or test file to generate predictions and your right submission file. In some cases, you must browse the test image folder to get the IDs and data.
13. The provided file paths are the only valid paths to load the data. Do not create any dummy data files.
14. **REPRODUCIBILITY**: Always use random_state=42 for ALL random operations (train_test_split, random sampling, etc.)
15. **SPLIT STRATEGY**: Use a single train/validation split only. DO NOT use k-fold or cross-validation.
16. **INPUT CONTRACT (MUST FOLLOW)**:
    - You MUST follow the "MODEL INPUT DATA CONTRACT" (if provided) and the input contract notes.
    - Ensure `preprocess_data(file_paths: dict)` returns exactly the expected artifact(s) 
17. **ABSOLUTE BAN: NO DUMMY / NO SYNTHETIC DATA (STRICT)**:
    - You MUST NOT generate, simulate, or fabricate ANY data under ANY condition (including "local testing only").
    - Do NOT add any code like:
      * `if not os.path.exists(...):` then create files/directories
      * `os.makedirs(...)` to construct a fake dataset
      * `pd.DataFrame(...).to_csv(...)` to create missing input CSVs
      * writing parquet/images/audio/video as placeholders
    - Only read from the provided dataset paths. Never write into the dataset directory.

## CODE STRUCTURE:
{code_structure_section}
"""

    def _default_code_structure(self, description: Dict[str, Any]) -> str:
        """Fallback code skeleton for preprocessing when PromptDecider does not provide one."""
        file_paths_main = description.get("link to the dataset", [])
        return (
            "```python\n"
            "# import necessary libraries\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "from sklearn.model_selection import train_test_split\n"
            "import sys\n"
            "import os\n"
            "\n"
            "def preprocess_data(file_paths: dict):\n"
            "    \"\"\"Preprocess data according to guidelines and contracts.\"\"\"\n"
            "    # Your preprocessing code here\n"
            "    return None\n"
            "\n"
            "if __name__ == \"__main__\":\n"
            "    try:\n"
            f"        file_paths = {repr(file_paths_main)}\n"
            "        _ = preprocess_data(file_paths)\n"
            "        print(\"Preprocessing finished.\")\n"
            "    except Exception as e:\n"
            "        print(f\"An error occurred during preprocessing test: {e}\", file=sys.stderr)\n"
            "        sys.exit(1)\n"
            "```\n"
        )

    def build(
        self,
        guideline: Dict,
        description: Dict,
        previous_code: str = None,
        error_message: str = None,
        iteration_type: str = None,
        input_contract_notes: list[str] | None = None,
        model_input_data: list[str] | None = None,
        datafile_structure: str | None = None,
        prompt_fields: Dict[str, Any] | None = None,
    ) -> str:
        """Build prompt to generate preprocessing code."""

        guideline = guideline or {}
        preprocessing_guideline = guideline.get('preprocessing', {})
        target_info = guideline.get("target_identification", {})

        prompt_fields = prompt_fields or {}

        # Add iteration-specific preprocessing guidance
        iteration_guidance = prompt_fields.get("iteration_guidance") or self._get_iteration_guidance(iteration_type)
        enhanced_guideline = json.dumps(preprocessing_guideline, indent=2)
        if iteration_guidance:
            enhanced_guideline += f"\n\n## ITERATION-SPECIFIC GUIDANCE:\n{iteration_guidance}"

        # Get batch processing instruction and data return format based on iteration
        batch_instruction, data_format = self._get_batch_processing_config(iteration_type)

        # Allow PromptDeciderAgent to override these placeholders
        batch_instruction = prompt_fields.get("batch_processing_instruction") or batch_instruction
        data_format = prompt_fields.get("data_return_format") or data_format

        input_contract_notes_section = prompt_fields.get("input_contract_notes_section") or ""
        if not input_contract_notes_section and input_contract_notes:
            lines = "\n".join(f"- {note}" for note in input_contract_notes if note)
            if lines:
                input_contract_notes_section = (
                    "## INPUT CONTRACT NOTES (from Knowledge Retrieval)\n"
                    + lines
                    + "\n"
                )

        model_input_data_section = prompt_fields.get("model_input_data_section") or ""
        if not model_input_data_section and model_input_data:
            items = [x for x in model_input_data if x]
            if items:
                model_input_data_section = (
                    "## MODEL INPUT DATA CONTRACT (from Knowledge Retrieval)\n"
                    "The modeling code MUST be able to consume the output of preprocess_data() in this form.\n"
                    + "\n".join(f"- {x}" for x in items)
                    + "\n"
                )

        prompt = self.template.format(
            dataset_name=description.get('name', 'N/A'),
            task_desc=description.get('task', 'N/A'),
            input_desc=description.get('input_data', ''),
            output_desc=description.get('output_data', ''),
            data_file_desc=json.dumps(description.get('data file description', {})),
            file_paths=description.get('link to the dataset', []),
            file_paths_main=description.get('link to the dataset', []),
            preprocessing_guideline=enhanced_guideline,
            target_info=json.dumps(target_info, indent=2),
            batch_processing_instruction=batch_instruction,
            data_return_format=data_format,
            input_contract_notes_section=input_contract_notes_section,
            model_input_data_section=model_input_data_section,
            code_structure_section=(prompt_fields.get("code_structure_section") if (prompt_fields or {}).get("code_structure_section") else self._default_code_structure(description)),
            datafile_structure=datafile_structure or "N/A",
        )

        if previous_code and error_message:
            # Use smart truncation for error message to save tokens and focus on relevant parts
            max_lines = getattr(self.manager.config, 'max_error_lines_for_llm', 20)
            max_chars = getattr(self.manager.config, 'max_error_message_length', 2048)
            truncated_error = smart_truncate_error(error_message, max_lines=max_lines, max_chars=max_chars)
            # Also include dataset paths from description analyzer in the retry context
            dataset_paths = description.get('link to the dataset', [])
            dataset_paths_json = json.dumps(dataset_paths)

            retry_context = f"""
## PREVIOUS ATTEMPT FAILED:
The previously generated code failed with an error.

### Previous Code:
```python
{previous_code}
```

### Error Message:
```
{truncated_error}
```

### Dataset Paths (from description analyzer):
{dataset_paths_json}

## FIX INSTRUCTIONS:
1. Analyze the error message and the previous code carefully.
2. Generate a new, complete, and corrected version of the Python code that resolves the issue.
3. Ensure the corrected code adheres to all the original requirements.
4. If the error indicates missing modules (ModuleNotFoundError/ImportError), wrap imports in try/except and, in the except block, use subprocess to install the missing package using only `["uv", "pip", "install", "--python", sys.executable, "<package>"]` with `check=True`, then retry the import.
5. Never use bare `pip`, `python`, `python3`, `conda`, `uv add`, shell install commands, or create/activate another virtual environment.

Generate the corrected Python code:
"""
            prompt += retry_context

        # Append full description analysis as JSON context
        try:
            description_json = json.dumps(description, indent=2, ensure_ascii=False)
        except Exception:
            description_json = json.dumps(description, indent=2)
        prompt += f"\n\n## FULL DESCRIPTION ANALYSIS (JSON)\n```json\n{description_json}\n```\n"

        self.manager.save_and_log_states(prompt, "preprocessing_coder_prompt.txt")
        return prompt
    
    def _get_iteration_guidance(self, iteration_type: str = None) -> str:
        """Get iteration-specific preprocessing guidance."""
        if iteration_type == "traditional":
            return """
For Traditional ML algorithms (XGBoost, LightGBM, CatBoost):
- Focus on feature engineering for tabular data
- **CRITICAL FOR IMAGE/VIDEO/AUDIO**: When extracting features from images/videos/audio using deep learning (CNNs, etc.), you MUST use batch-by-batch processing:
  * Load image paths/IDs first, NOT the actual images
  * Use a loop with batch_size (32-128): for i in range(0, len(data), batch_size)
  * For each batch: load images → extract features → append to list → clear images from memory
  * After all batches: concatenate feature lists
  * DO NOT: create a single large numpy array of all images before feature extraction
- Use categorical encoding (Label/One-hot/Target encoding)
- Apply numerical feature scaling if needed (StandardScaler, MinMaxScaler)
- Handle missing values with appropriate imputation strategies
- Consider feature selection techniques (SelectKBest, RFE)
- Ensure all features are numerical for tree-based models
- Load the resulting compact tabular features (after extraction) into memory for training
"""
        elif iteration_type == "custom_nn":
            return """
For Custom Neural Networks:
- Apply numerical normalization/standardization (StandardScaler, MinMaxScaler)
- Convert categorical variables to numerical embeddings or one-hot encoding
- Reshape data to proper format for neural network input
- Create train/validation splits suitable for NN training with proper batching
- Apply data augmentation techniques if applicable
- Ensure consistent data types (float32/float64)
- Consider dimensionality reduction if needed
"""
        elif iteration_type == "custom_nn_search":
            return """
For Custom Neural Networks with Architecture Search:
- Apply numerical normalization/standardization (StandardScaler, MinMaxScaler)
- Convert categorical variables to numerical embeddings or one-hot encoding
- Reshape data to proper format based on the suggested architecture pattern
- Consider the architecture structure when preparing input dimensions
- Create train/validation splits suitable for NN training with proper batching
- Apply data augmentation techniques if applicable
- Ensure consistent data types (float32/float64)
- Consider dimensionality reduction if needed based on architecture requirements
"""
        elif iteration_type == "pretrained":
            return """
For Pretrained Models (prioritize PyTorch over TensorFlow when possible):
- Format data to match the chosen pretrained model’s input requirements.
- Follow the MODEL INPUT DATA CONTRACT (if provided) and input contract notes.
- For text: tokenize using the chosen model’s tokenizer/processor; output tensors/datasets compatible with the modeling code.
- For images/video/audio: apply model-specific resizing/normalization/feature extraction; keep batch-by-batch processing to avoid OOM.
- For tabular: return clean numerical tensors/arrays; preserve consistent column order across train/val/test.
- Ensure preprocessing matches the pretrained model’s expected training/inference format (special tokens, padding, normalization, etc.).
"""
        else:
            return ""
    
    def _get_batch_processing_config(self, iteration_type: str = None) -> tuple[str, str]:
        """Get batch processing instruction and data return format based on iteration type."""
        if iteration_type == "traditional":
            batch_instruction = "IMPORTANT: Load entire dataset into memory for traditional ML algorithms."
            data_format = "5. Create a function `preprocess_data()` that takes a dictionary of file paths and returns a tuple of **preprocessed DataFrames/arrays** (e.g., X_train, X_val, X_test, y_train, y_val, y_test)."
        elif iteration_type == "custom_nn":
            batch_instruction = "IMPORTANT: Preprocess data by batch using generators to reduce memory usage for neural network training."
            data_format = "5. Create a function `preprocess_data()` that takes a dictionary of file paths and returns a tuple of **generators**, one for each data split (e.g., train_generator, val_generator, test_generator)."
        elif iteration_type == "pretrained":
            batch_instruction = (
                "IMPORTANT: Return the most appropriate input artifacts for the chosen pretrained model. "
                "If a MODEL INPUT DATA CONTRACT is provided, you MUST follow it exactly. "
                "Otherwise, choose the simplest artifact that the modeling code can consume reliably (e.g., in-memory arrays, finite DataLoaders, or a dataset config path)."
            )
            data_format = (
                "5. Create a function `preprocess_data()` that takes a dictionary of file paths and returns the most appropriate "
                "input artifacts for the chosen pretrained model. If a MODEL INPUT DATA CONTRACT is provided, you MUST follow it exactly."
            )
        else:
            # Default behavior
            batch_instruction = "IMPORTANT: Preprocess data by batch using generators to reduce memory usage."
            data_format = "5. Create a function `preprocess_data()` that takes a dictionary of file paths and returns a tuple of **generators**, one for each data split (e.g., train_generator, val_generator, test_generator)."
        
        return batch_instruction, data_format

    def parse(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response
        
        self.manager.save_and_log_states(code, "preprocessing_code_response.py")
        return code
