# src/iML/prompts/guideline_prompt.py
import json
import logging
import re
from typing import Any, Dict, Optional

from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


def _strip_bom(s: str) -> str:
    if s.startswith("\ufeff"):
        return s[1:]
    return s


def _strip_ansi_and_osc(s: str) -> str:
    """Remove common ANSI/OSC sequences that can break json.loads if echoed into the reply."""
    s = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", s)
    s = re.sub(r"\x1b\][^\x07]*(?:\x07|\x1b\\)", "", s)
    return s


def _extract_fenced_json(s: str) -> str:
    """Prefer ```json ... ``` or first ``` ... ``` block that looks like JSON."""
    s = s.strip()
    if "```json" in s:
        inner = s.split("```json", 1)[1]
        if "```" in inner:
            return inner.split("```", 1)[0].strip()
    if "```" in s:
        chunks = s.split("```")
        for i in range(1, len(chunks), 2):
            block = chunks[i].strip()
            if block.lower().startswith("json"):
                block = block[4:].lstrip("\n").strip()
            if "{" in block:
                return block
    return s


def _iter_balanced_json_objects(s: str):
    """Yield top-level {...} substrings with balanced braces (string-aware)."""
    for start, ch in enumerate(s):
        if ch != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(s)):
            c = s[i]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield s[start : i + 1]
                    break


def _looks_like_guideline_json(chunk: str) -> bool:
    if not chunk or not isinstance(chunk, str):
        return False
    # Avoid false positives such as "{id}" or format placeholders inside prose.
    if '":' not in chunk:
        return False
    expected_keys = [
        '"target_identification"',
        '"modeling"',
        '"preprocessing"',
    ]
    return any(k in chunk for k in expected_keys)


def _extract_balanced_json_object(s: str) -> Optional[str]:
    """Return the first plausible guideline JSON object from arbitrary text."""
    for chunk in _iter_balanced_json_objects(s):
        if _looks_like_guideline_json(chunk):
            return chunk
    return None


def _normalize_llm_json_text(response: str) -> str:
    if not response or not str(response).strip():
        return ""
    s = _strip_bom(str(response).strip())
    s = _strip_ansi_and_osc(s)
    s = _extract_fenced_json(s)
    s = s.strip()
    if s.startswith("{") and _looks_like_guideline_json(s):
        return s
    if not s.startswith("{") or not _looks_like_guideline_json(s):
        extracted = _extract_balanced_json_object(s)
        if extracted:
            s = extracted
    return s

def _create_variables_summary(variables: dict) -> dict:
    """Create a concise summary for variables in the profile."""
    summary = {}
    for var_name, var_details in variables.items():
        summary[var_name] = {
            "type": var_details.get("type"),
            "n_unique": var_details.get("n_unique"),
            "p_missing": var_details.get("p_missing"),
            "mean": var_details.get("mean"),
            "std": var_details.get("std"),
            "min": var_details.get("min"),
            "max": var_details.get("max"),
        }
    return summary

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
- Size: {n_rows:,} rows, {n_cols} columns
- Key Quality Alerts: {alerts}
- Output format: {output_data}
- Submission file description: {submission_file_description}

## Variables Analysis Summary:
```json
{variables_summary_str}
```

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


Output Format: Your response must be a SINGLE valid JSON object matching the shape below.
IMPORTANT: Ensure the generated JSON is perfectly valid.
- All strings must be enclosed in double quotes.
- All backslashes inside strings must be properly escaped.
- There should be no unescaped newline characters within a string value.
- Do not include comments within the JSON output.
- Do not include markdown fences.
- Do not include any prose before or after the JSON object.

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
        "IDs_in_submission_file_contain_file_extensions": true,
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
        {{
            "step": 3,
            "action": "data_splitting",
            "train_size": 0.8,
            "validation_size": 0.2,
            "strategy": "simple_random",
            "random_state": 42,
            "notes": "split data into train and validation sets"
        }}
    ]
}}

CRITICAL FINAL REMINDER:
- Return JSON only.
- If you are about to explain your answer in prose, do not do that; put the content into the JSON fields instead.
""" 

    def build(
        self,
        description_analysis: Dict[str, Any],
        profiling_result: Dict[str, Any],
        model_suggestions: Dict[str, Any] | None = None,
        iteration_type: str | None = None,
        task_context: Dict[str, Any] | None = None,
        knowledge_pack: Dict[str, Any] | None = None,
        datafile_structure: str | None = None,
    ) -> str:
        """Build prompt from analysis and profiling results.

        Supports two formats:
        - Summarized profiling (preferred): keys include 'files', 'label_analysis', 'feature_quality'.
        - Raw profiling (fallback): keys include 'summaries', 'profiles'.
        """
        task_info = description_analysis

        dataset_name = task_info.get('name', 'N/A')
        task_desc = task_info.get('task', 'N/A')
        output_data = task_info.get('output_data', 'N/A')
        submission_file_description = task_info.get('submission file description', 'N/A')
        n_rows = 0
        n_cols = 0
        alerts_out = []
        variables_summary_dict = {}

        if 'label_analysis' in profiling_result or 'files' in profiling_result:
            # Summarized format
            files = profiling_result.get('files', []) or []
            # choose train-like file if present
            chosen = None
            for f in files:
                name = (f.get('name') or '').lower()
                if 'train' in name and 'test' not in name and 'submission' not in name:
                    chosen = f
                    break
            if not chosen and files:
                chosen = files[0]
            if chosen:
                n_rows = chosen.get('n_rows', 0) or 0
                n_cols = chosen.get('n_cols', 0) or 0

            la = profiling_result.get('label_analysis', {}) or {}
            fq = profiling_result.get('feature_quality', {}) or {}

            # alerts: concise messages
            if la:
                if la.get('has_label_column') is False:
                    alerts_out.append('No label column detected')
                if la.get('has_missing_labels'):
                    alerts_out.append('Missing labels present')
                imb = la.get('class_distribution_imbalance')
                if imb and imb != 'none':
                    alerts_out.append(f'label imbalance: {imb}')
                if la.get('num_classes'):
                    alerts_out.append(f"num_classes={la['num_classes']}")

            if fq:
                hm = fq.get('high_missing_columns') or []
                if hm:
                    alerts_out.append(f"high-missing cols: {len(hm)}")
                hc = fq.get('high_cardinality_categoricals') or []
                if hc:
                    alerts_out.append(f"high-cardinality cats: {len(hc)}")

            # variables summary minimal to avoid noise
            variables_summary_dict = {
                'high_missing_columns': fq.get('high_missing_columns') or [],
                'high_cardinality_categoricals': fq.get('high_cardinality_categoricals') or [],
                'date_like_cols': fq.get('date_like_cols') or [],
                'label_column': la.get('label_column'),
            }
        else:
            # Fallback to raw profiling (legacy)
            train_key = None
            for key in profiling_result.get('summaries', {}).keys():
                if 'test' not in key.lower() and 'submission' not in key.lower():
                    train_key = key
                    break
            if not train_key:
                train_key = next(iter(profiling_result.get('summaries', {})), None)

            train_summary = profiling_result.get('summaries', {}).get(train_key, {})
            train_profile = profiling_result.get('profiles', {}).get(train_key, {})
            n_rows = train_summary.get('n_rows', 0)
            n_cols = train_summary.get('n_cols', 0)
            alerts = train_profile.get('alerts', [])
            variables = train_profile.get('variables', {})
            alerts_out = alerts[:3] if alerts else []
            variables_summary_dict = _create_variables_summary(variables)

        # Build auxiliary sections
        variables_summary_str = json.dumps(variables_summary_dict, indent=2, ensure_ascii=False)
        model_suggestions = model_suggestions or {}
        # Extract SOTA models (from ADK) if present
        sota_models = model_suggestions.get('sota_models', []) or []
        model_suggestions_str = json.dumps(model_suggestions, indent=2, ensure_ascii=False)

        # Generate ID format section
        id_format_section = self._generate_id_format_section(profiling_result)

        # Generate algorithm constraint based on iteration type
        algorithm_constraint = self._get_algorithm_constraint(iteration_type)

        # If pretrained iteration and SOTA models exist, add a hard requirement block
        sota_section = ""
        if iteration_type == "pretrained" and sota_models:
            # Keep only lightweight view for the prompt
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
        
        # If custom_nn_search iteration and architecture suggestions exist, add architecture guidance
        architecture_section = ""
        architecture_suggestions = getattr(self.manager, "architecture_suggestions", None)
        if iteration_type == "custom_nn_search" and architecture_suggestions:
            architectures = architecture_suggestions.get("architectures", [])
            if architectures:
                # Keep only architecture info for the prompt
                arch_shortlist = [
                    {
                        "architecture_name": a.get("architecture_name"),
                        "architecture_structure": a.get("architecture_structure"),
                        "source_link": a.get("source_link"),
                    }
                    for a in architectures[:1]  # Only first candidate
                ]
                architecture_section = (
                    "\n## SUGGESTED NEURAL NETWORK ARCHITECTURE (from search)\n"
                    + json.dumps(arch_shortlist, indent=2, ensure_ascii=False)
                    + "\n\nRECOMMENDATION (CUSTOM_NN_SEARCH): You should adapt the suggested architecture pattern above for this specific task.\n"
                      "Feel free to modify layer sizes, add/remove layers, or adjust hyperparameters based on the data characteristics.\n"
                      "The architecture structure should guide your design, but you have flexibility to optimize it for this problem."
                )

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

        datafile_structure = datafile_structure or profiling_result.get("directory_structure_summary") or "N/A"

        prompt = self.template.format(
            dataset_name=dataset_name,
            task_desc=task_desc,
            n_rows=n_rows,
            n_cols=n_cols,
            alerts=alerts_out if alerts_out else 'None',
            variables_summary_str=variables_summary_str,
            output_data=output_data,
            submission_file_description=submission_file_description,
            model_suggestions_str=model_suggestions_str,
            algorithm_constraint=algorithm_constraint,
            id_format_section=id_format_section + sota_section + architecture_section,
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

    def _generate_id_format_section(self, profiling_result: Dict[str, Any]) -> str:
        """Generate ID format analysis section for the prompt."""
        # Check if we have ID format analysis
        id_format_analysis = profiling_result.get('id_format_analysis', {})
        
        if not id_format_analysis:
            return ""
        
        has_extensions = id_format_analysis.get('has_file_extensions', False)
        detected_extensions = id_format_analysis.get('detected_extensions', [])
        format_notes = id_format_analysis.get('format_notes', [])
        submission_analysis = id_format_analysis.get('submission_format_analysis')
        
        if not has_extensions and not format_notes:
            return ""
        
        section_lines = ["## ID FORMAT ANALYSIS:"]
        
        if has_extensions:
            section_lines.append(f"- **ID columns contain file extensions**: {', '.join(detected_extensions)}")
        
        if submission_analysis:
            submission_has_ext = submission_analysis.get('submission_has_extensions', False)
            submission_file = submission_analysis.get('submission_file', 'N/A')
            section_lines.append(f"- **Submission format detected**: File extensions {'required' if submission_has_ext else 'NOT required'} in {submission_file}")
        
        if format_notes:
            section_lines.append("- **CRITICAL NOTES**:")
            for note in format_notes:
                section_lines.append(f"  * {note}")
        
        section_lines.extend([
            "",
            "**PREPROCESSING NOTE**: If ID format mismatch detected, ensure preprocessing handles ID transformation correctly.",
            "**MODELING NOTE**: You MUST set 'IDs_in_submission_file_contain_file_extensions' field in the modeling section based on the 'Submission format detected' above. When creating submission files, ensure ID format matches exactly (with or without file extensions).",
            ""
        ])
        
        return "\n".join(section_lines)

    def parse(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM (tolerates markdown fences, leading prose, minor noise)."""
        normalized = _normalize_llm_json_text(response or "")
        if not normalized:
            logger.error("Failed to parse JSON from LLM response for guideline: empty or whitespace-only response")
            parsed_response = {
                "error": "Empty or non-JSON response from LLM",
                "raw_response": response,
            }
        else:
            try:
                parsed_response = json.loads(normalized)
            except json.JSONDecodeError as e:
                # Last resort: balanced {...} inside normalized (handles trailing junk)
                fallback = _extract_balanced_json_object(normalized)
                if fallback and fallback != normalized:
                    try:
                        parsed_response = json.loads(fallback)
                    except json.JSONDecodeError as e2:
                        logger.error(
                            "Failed to parse JSON from LLM response for guideline: %s (fallback also failed: %s)",
                            e,
                            e2,
                        )
                        parsed_response = {
                            "error": "Invalid JSON response from LLM",
                            "raw_response": response,
                            "normalized_head": normalized[:4000],
                        }
                else:
                    logger.error(f"Failed to parse JSON from LLM response for guideline: {e}")
                    parsed_response = {
                        "error": "Invalid JSON response from LLM",
                        "raw_response": response,
                        "normalized_head": normalized[:4000],
                    }
            if isinstance(parsed_response, dict) and "error" not in parsed_response:
                required = {"target_identification", "modeling", "preprocessing"}
                if not required.issubset(set(parsed_response.keys())):
                    parsed_response = {
                        "error": "Guideline JSON missing required top-level keys",
                        "raw_response": response,
                        "normalized_head": normalized[:4000],
                    }
        
        self.manager.save_and_log_states(
            json.dumps(parsed_response, indent=4, ensure_ascii=False), 
            "guideline/guideline_response.json"
        )
        return parsed_response
