# src/iML/prompts/description_analyzer_prompt.py
import json
import os
from .base_prompt import BasePrompt
from typing import Dict, Any 

class DescriptionAnalyzerPrompt(BasePrompt):

    def default_template(self) -> str:
        return """
You are an expert AI assistant specializing in analyzing Kaggle competition descriptions. Your task is to read the provided text and extract key information into a specific JSON structure.
The output MUST be a valid JSON object and nothing else. Do not include any explanatory text before or after the JSON.
Extract the following information:
- "name": Dataset name
- "input_data": A description of the primary input data for the model.
- "output_data": A description of the expected output format from the model.
- "task": A summary of the main objective or task of the competition.
- "task_type": One of ["text_classification","image_classification","tabular_classification","tabular_regression","seq2seq","ner","qa","unknown"] inferred from the description and directory structure.
- "data file description": A dictionary where the keys are relative path to the file (e.g., "train.csv", "test.csv") and the values are their descriptions. (only the file or folder name, not the father directory)
- "submission file description": A dictionary describing the submission file columns (column name -> meaning).
- "eval_metrics": The evaluation metric for this task (get from description; if not found choose an appropriate metric).
- "link to the dataset": A list containing the filenames and folders of the core data files (like train, test, sample submission).
  IMPORTANT: Use the DATA FILE STRUCTURE section as ground truth. Only return paths that exist there.
  Return RELATIVE paths from the input_data_folder only: do NOT include the input_data_folder prefix and do NOT include the root folder name shown at the top of the tree.
## EXAMPLE:
### INPUT TEXT:

\"\"\"
Welcome to the 'Paddy Disease Classification' challenge! The goal is to classify diseases in rice paddy images. The input data consists of images of rice plants (JPG files) from the `train_images` folder. Your model should output a class label for one of ten possible diseases. The dataset includes `train.csv` which maps image IDs to their labels, `test_images` for prediction, and `sample_submission.csv` for the required format.
\"\"\"

### OUTPUT JSON:

{{
    "name": "paddy_disease_classification",
    "input_data": "The input data consists of images of rice plants (JPG files).",
    "output_data": "The model should output a class label corresponding to one of ten possible diseases.",
    "task": "The main goal is to build a model that can classify diseases in rice paddy images.",
    "task_type": "image_classification",
    "data file description": {{
        "train.csv": "Maps image IDs to their respective disease labels.",
        "train_images": "A folder containing the training images as JPG files.",
        "test_images": "A folder containing the test images for which predictions are required.",
        "sample_submission.csv": "An example file showing the required submission format."
    }},
    "submission file description": {{
        "image_id": "Identifier for the image (matches filenames or IDs in test set).",
        "label": "Predicted class label."
    }},
    "eval_metrics": "accuracy",
    "link to the dataset": ["train.csv", "train_images", "test_images", "sample_submission.csv"]
}}
## END OF EXAMPLE. NOW, PROCESS THE FOLLOWING TEXT:
### INPUT TEXT:
\"\"\"
### DESCRIPTION.TXT CONTENT
{description}

### DATA FILE STRUCTURE (auto-extracted from input_data_folder; do NOT invent paths)
{directory_structure}
\"\"\"
### OUTPUT JSON:
"""

    # Fixed build method to be correct
    def build(self, description: str, directory_structure: str) -> str:
        """
        Build complete prompt from template and input values.
        """

        prompt = self.template.format(
            description=description,
            directory_structure=directory_structure
        )
        self.manager.save_and_log_states(
            content=prompt, 
            save_name="description_analyzer_prompt.txt"
        )
        return prompt

    # Fixed parse method to be correct
    def parse(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract JSON object.
        """
        try:
            # Clean and parse JSON string from response
            clean_response = response.strip().replace("```json", "").replace("```", "")
            parsed_response = json.loads(clean_response)
        except json.JSONDecodeError as e:
            # dùng logging toàn cục nếu bạn đã bỏ self.manager.logger
            try:
                import logging
                logging.error(f"Failed to parse JSON from LLM response: {e}")
            except Exception:
                pass
            parsed_response = {"error": "Invalid JSON response from LLM", "raw_response": response}

        # Normalize and convert link_to_dataset entries to absolute paths for downstream tooling
        # (ProfilingAgent expects filesystem-valid paths). The LLM is instructed to output relative paths.
        if "link to the dataset" in parsed_response and isinstance(parsed_response["link to the dataset"], list):
            dataset_path = (self.manager.input_data_folder or "").replace("\\", "/").rstrip("/")
            root_name = os.path.basename(dataset_path) if dataset_path else ""

            rel_items: list[str] = []
            for raw in parsed_response.get("link to the dataset", []) or []:
                if raw is None:
                    continue
                s = str(raw).strip().replace("\\", "/")
                if not s:
                    continue

                # Strip accidental absolute prefix
                if dataset_path and s.startswith(dataset_path + "/"):
                    s = s[len(dataset_path) + 1 :]

                # Strip accidental root folder name (tree usually starts with "<root_name>/")
                if root_name and (s == root_name or s.startswith(root_name + "/")):
                    s = s[len(root_name) + 1 :] if s.startswith(root_name + "/") else ""

                s = s.lstrip("./").lstrip("/")
                if not s:
                    continue
                rel_items.append(s)

            # De-duplicate while preserving order
            seen = set()
            rel_items_dedup: list[str] = []
            for s in rel_items:
                if s in seen:
                    continue
                seen.add(s)
                rel_items_dedup.append(s)

            parsed_response["link_to_dataset_rel"] = rel_items_dedup
            if dataset_path:
                parsed_response["link to the dataset"] = [
                    os.path.join(dataset_path, fname).replace("\\", "/") for fname in rel_items_dedup
                ]
            else:
                parsed_response["link to the dataset"] = rel_items_dedup

        self.manager.save_and_log_states(
            content=json.dumps(parsed_response, indent=2), 
            save_name="description_analyzer_response.json"
        )
        return parsed_response
