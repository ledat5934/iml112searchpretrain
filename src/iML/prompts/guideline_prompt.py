# src/iML/prompts/guideline_prompt.py
import json
import logging
from typing import Dict, Any

from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


class GuidelinePrompt(BasePrompt):
    """
    Prompt handler to create guidelines for AutoML pipeline.
    """

    def default_template(self) -> str:
        """Default template to request LLM to create guidelines."""
        return """You are an expert Machine Learning architect. Your task is to analyze the provided dataset information and create a specific, actionable, and justified guideline for an machine learning pipeline.
        Using recent knowledge and state-of-the-art studies to devise promising high-quality plan
## Dataset Information:
- Dataset: {dataset_name}
- Task: {task_desc}
- Output format: {output_data}
- Submission file description: {submission_file_description}

## DATAFILE STRUCTURE (SUMMARY)
{datafile_structure}

{task_context_section}

{knowledge_section}

{id_format_section}

## IMPORTANT CONSTRAINTS:
- ALWAYS use random_state=42 for ALL random operations (train_test_split, model initialization)
- Use a single holdout split for train/validation (NO k-fold or cross-validation)
- Ensure that your plan is up-to-date with current state-of-the-art knowledge.
- Ensure that your plan is designed for AI agents coders instead of human engineers.
- Ensure that your plan is self-contained with sufficient instructions to be executed by the AI agents. 
- Ensure that your plan includes all the key points and instructions (from handling data to modeling) so that the AI agents can successfully implement them. Do NOT directly write the code.
- Ensure that your plan completely include the end-to-end process of machine learning pipeline in detail (i.e., from data loading to model training and submission creation) when applicable based on the given requirements.
- **INPUT CONTRACT (CRITICAL)**: If a knowledge pack is provided, you MUST align your plan with `knowledge_pack.model_input_data` and `knowledge_pack.input_contract_notes`.
  - Explicitly state what `preprocess_data()` must return (e.g., DataFrames/arrays vs DataLoaders vs a config path like `data.yaml`) and how the modeling code will consume it.
- **CRITICAL MEMORY CONSTRAINT FOR NEURAL NETWORKS**: When using neural networks (custom NN or pretrained models) with image/video/audio data or large datasets, you MUST specify batch processing approach in your preprocessing strategy. Use batch_size (e.g., 32, 64, 128) for feature extraction, data loading, and prediction. For traditional ML algorithms, you can load entire preprocessed features into memory after feature extraction is done in batches.

JUSTIFY YOUR CHOICES INTERNALLY: Even if the final JSON does not include every reasoning detail, your internal decision process must be sound, based on the data properties.


{algorithm_constraint}

Before generating the final JSON, consider:
1. Identify the target variable and task type (classification, regression, etc.).
2. Review each variable's type, statistics, and potential issues.
3. Choose appropriate and reasonable preprocessing steps for that algorithm type.
4. **For image/video/audio data with traditional ML**: You MUST specify explicit batch-by-batch processing to prevent memory overflow. Example strategy_or_details: "Use batch_size=64 for feature extraction. Load image paths/IDs first (not the actual images). Process images in batches using a loop: for each batch, load only that batch of images into memory, extract features using pre-trained CNN (EfficientNetB0), store features, then clear images from memory. After processing all batches, concatenate all extracted features. The resulting compact tabular features can then be loaded into memory for traditional ML training. DO NOT load all images into a single numpy array before feature extraction."
5. **For image/video/audio data with neural networks**: Use batch processing throughout (data loading, training, prediction) with generators. Example: "Use batch_size=32 for data loading and training with generators"
6. Compile these specific actions into the required JSON format.


Output Format: Your response must be in the JSON format below:
IMPORTANT: Ensure the generated JSON is perfectly valid.
- All strings must be enclosed in double quotes.
- All backslashes inside strings must be properly escaped.
- There should be no unescaped newline characters within a string value.
- Do not include comments within the JSON output.

{{
    "target_identification": {{
        "target_variable": "identified_target_column_name",
        "reasoning": "Explanation for target selection based on submission file and task.",
        "task_type": "classification/regression/etc"
    }},
    "modeling": {{
        "recommended_algorithms": ["one most suitable algorithm"],
        "model_selection": ["model_name"],
        "eval_metrics": ["metric"],
        "random_state": 42,
        "notes": "additional notes about model selection and training",
        "IDs_in_submission_file_contain_file_extensions": true/false (MUST match the 'Submission format detected' from ID FORMAT ANALYSIS section above. If submission requires file extensions, set true; otherwise false. If no ID FORMAT ANALYSIS provided, infer from submission file description),
        "training_strategy": {{
            "approach": "SOTA training approach and techniques"
        }},
        "create_submission_file": {{
            "guideline": "guideline to create submission file. CRITICAL: Ensure IDs in submission match the format specified in IDs_in_submission_file_contain_file_extensions field (with or without file extensions like .jpg, .png, .mp4, etc.)",
            "notes": "additional notes about submission file creation"
        }}
    }},
    "preprocessing": [
        {{
            "step": 1,
            "action": "action_type (e.g., impute_missing, encode_categorical, scale_numerical, feature_engineering, drop_columns, clean_data)",
            "columns": ["column_name_1", "column_name_2"],
            "strategy_or_details": "e.g., 'median', 'one_hot_encoder', 'standard_scaler', 'NewFeature = ColA / ColB', 'drop_reason'"
        }},
        {{
            "step": 2,
            "action": "...",
            "columns": ["..."],
            "strategy_or_details": "..."
        }},
        #Add some more preprocessing step here if neccessary.
        {{
            "step": ,
            "action": "data_splitting",
            "train_size": 0.8,
            "validation_size": 0.2,
            "strategy": "simple_random",
            "random_state": 42,
            "notes": "split data into train and validation sets"
        }}
    ]
}}"""

    def build(
        self,
        description_analysis: Dict[str, Any],
        profiling_result: Dict[str, Any] | None = None,
        model_suggestions: Dict[str, Any] | None = None,
        iteration_type: str | None = None,
        task_context: Dict[str, Any] | None = None,
        knowledge_pack: Dict[str, Any] | None = None,
        datafile_structure: str | None = None,
    ) -> str:
        """Build prompt from description analysis and optional context."""
        task_info = description_analysis

        dataset_name = task_info.get('name', 'N/A')
        task_desc = task_info.get('task', 'N/A')
        output_data = task_info.get('output_data', 'N/A')
        submission_file_description = task_info.get('submission file description', 'N/A')

        model_suggestions = model_suggestions or {}
        sota_models = model_suggestions.get('sota_models', []) or []

        algorithm_constraint = self._get_algorithm_constraint(iteration_type)

        # SOTA model shortlist for pretrained iterations
        sota_section = ""
        if iteration_type == "pretrained" and sota_models:
            shortlist = [
                {
                    "model_name": m.get("model_name"),
                    "model_link": m.get("model_link"),
                }
                for m in sota_models[:10]
            ]
            sota_section = (
                "\n## SOTA MODEL SHORTLIST (from ADK search)\n"
                + json.dumps(shortlist, indent=2, ensure_ascii=False)
                + "\n\nIMPORTANT (PRETRAINED): You MUST choose a model from the SOTA shortlist above (or its exact HF model) for 'model_selection'.\n"
                  "Provide configuration aligned with the chosen model."
            )
        
        # Architecture guidance for custom_nn_search iterations
        architecture_section = ""
        architecture_suggestions = getattr(self.manager, "architecture_suggestions", None)
        if iteration_type == "custom_nn_search" and architecture_suggestions:
            architectures = architecture_suggestions.get("architectures", [])
            if architectures:
                arch_shortlist = [
                    {
                        "architecture_name": a.get("architecture_name"),
                        "architecture_structure": a.get("architecture_structure"),
                        "source_link": a.get("source_link"),
                    }
                    for a in architectures[:1]
                ]
                architecture_section = (
                    "\n## SUGGESTED NEURAL NETWORK ARCHITECTURE (from search)\n"
                    + json.dumps(arch_shortlist, indent=2, ensure_ascii=False)
                    + "\n\nRECOMMENDATION (CUSTOM_NN_SEARCH): You should adapt the suggested architecture pattern above for this specific task.\n"
                      "Feel free to modify layer sizes, add/remove layers, or adjust hyperparameters based on the data characteristics.\n"
                      "The architecture structure should guide your design, but you have flexibility to optimize it for this problem."
                )

        id_format_section = sota_section + architecture_section

        task_context_section = ""
        if task_context:
            task_context_section = (
                "## TASK CONTEXT (DESCRIPTION + SCHEMA)\n"
                + "```json\n"
                + json.dumps(task_context, indent=2, ensure_ascii=False)
                + "\n```\n"
            )

        knowledge_section = ""
        if knowledge_pack:
            knowledge_section = (
                "## ITERATION-SPECIFIC KNOWLEDGE PACK (NO EXAMPLE CODE)\n"
                + "```json\n"
                + json.dumps(knowledge_pack, indent=2, ensure_ascii=False)
                + "\n```\n"
            )

        datafile_structure = datafile_structure or "N/A"

        prompt = self.template.format(
            dataset_name=dataset_name,
            task_desc=task_desc,
            output_data=output_data,
            submission_file_description=submission_file_description,
            algorithm_constraint=algorithm_constraint,
            id_format_section=id_format_section,
            task_context_section=task_context_section,
            knowledge_section=knowledge_section,
            datafile_structure=datafile_structure,
        )

        self.manager.save_and_log_states(prompt, "guideline/guideline_prompt.txt")
        return prompt
    
    def _get_algorithm_constraint(self, iteration_type: str | None) -> str:
        """Get algorithm constraint based on iteration type."""
        if iteration_type == "traditional":
            return "IMPORTANT: YOU MUST USE TRADITIONAL ML ALGORITHMS: XGBoost, LightGBM, CatBoost, Linear regression, SVM, Bayes, TabPFN, ..."
        elif iteration_type == "custom_nn":
            return "IMPORTANT: YOU MUST BUILD CUSTOM NEURAL NETWORKS from scratch using PyTorch. "
        elif iteration_type == "custom_nn_search":
            return "IMPORTANT: YOU MUST BUILD CUSTOM NEURAL NETWORKS from scratch using PyTorch, following the suggested architecture pattern below."
        elif iteration_type == "pretrained":
            return "IMPORTANT: YOU MUST USE PRETRAINED MODELS"
        else:
            # Default for backward compatibility
            return "None"

    def parse(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        try:
            parsed_response = json.loads(response.strip().replace("```json", "").replace("```", ""))
        except json.JSONDecodeError as e:
            # Use local logger; manager.logger may not exist in some environments/checkpoints
            logger.error(f"Failed to parse JSON from LLM response for guideline: {e}")
            parsed_response = {"error": "Invalid JSON response from LLM", "raw_response": response}
        
        self.manager.save_and_log_states(
            json.dumps(parsed_response, indent=4, ensure_ascii=False), 
            "guideline/guideline_response.json"
        )
        return parsed_response
