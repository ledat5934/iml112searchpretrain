# src/iML/prompts/knowledge_retrieval_prompt.py
import json
from typing import Dict, Any

from .base_prompt import BasePrompt


class KnowledgeRetrievalPrompt(BasePrompt):
    """
    Prompt handler to synthesize iteration-specific knowledge packs.
    """

    def default_template(self) -> str:
        return """You are an expert ML engineer. Produce an ITERATION-SPECIFIC KNOWLEDGE PACK to guide planning.
This knowledge pack should summarize best practices and pitfalls for the given task context and iteration type, and appropriate with model chosen.
You MUST pay special attention to FEATURE ENGINEERING / PREPROCESSING TRANSFORMS that are known to work well for similar tasks.

ITERATION TYPE: {iteration_type}

## TASK CONTEXT (JSON)
```json
{task_context_json}
```

## MODEL SHORTLIST (JSON) - optional
```json
{model_suggestions_json}
```
(Your knowledge pack should be appropriate with the model chosen)
## ARCHITECTURE PATTERNS (JSON) - optional
```json
{architecture_suggestions_json}
```
(Your knowledge pack should be appropriate with the architecture chosen)
## OUTPUT REQUIREMENTS
If you have access to web search (e.g., google_search tool), you SHOULD:
- run 2–5 focused searches about feature engineering / preprocessing best practices for this task type and domain,
- read from multiple sources (papers, blogs, Kaggle discussions),
- and then synthesize the insights into the knowledge pack (without copying code).

Return a SINGLE JSON object with the following fields (add more if needed, but keep it concise):
{{
  "iteration_type": "{iteration_type}",
  "preprocessing_recommendations": [
    {{"step": "...", "details": "..."}}
  ],
  "training_recommendations": [
    {{"item": "...", "details": "..."}}
  ],
  "validation_and_metrics": {{
    "metrics": ["..."],
    "split_strategy": "...",
    "notes": "..."
  }},
  "feature_engineering": {{
    "global_principles": [
      "High-level rules for feature engineering on this task type (e.g., leakage avoidance, temporal ordering, handling rare categories)."
    ],
    "feature_templates": [
      {{"name": "...", "pattern": "What kind of feature (e.g., time-based aggregation, interaction term, text length, count encoding).", "when_to_use": "...", "caveats": "..." }}
    ],
    "task_specific_ideas": [
      "Concrete feature ideas tailored to THIS dataset/task description (e.g., for PetFinder: age buckets, description length, photo count bins, rescue type indicators)."
    ],
    "leakage_risks": [
      "Feature-level leakage warnings (e.g., avoid using future timestamps, avoid aggregations that peek into validation/test)."
    ]
  }},
  "model_input_data": [
    "dataframe|numpy_arrays|dataloader|data_yaml_path|hf_dataset|file_paths_only|..."(only output 1)
  ],
  "input_contract_notes": [
    "Short, concrete constraints about what preprocess_data() MUST return to be consumable by the chosen model/framework."
  ],
  "submission_notes": [
    "..."
  ],
  "common_pitfalls": [
    "..."
  ],
  "quality_checklist": [
    "..."
  ],
  "sources_or_keywords": [
    "Short list of URLs or search keywords that informed this knowledge pack."
  ],
  "notes": "This pack is guidance only; do not include example code."
}}

IMPORTANT:
- Output MUST be valid JSON (no markdown, no code fences).
- Do NOT include any example code.
- `model_input_data` is a short contract for downstream agents:
  - Use one PRIMARY value (first item) and optionally 1-2 secondary alternatives.
  - Examples of intended meaning:
    - "dataframe": preprocessing returns in-memory pandas DataFrames/arrays.
    - "dataloader": preprocessing returns PyTorch DataLoaders / Dataset objects.
    - "data_yaml_path": preprocessing writes a `data.yaml` file and returns its path for frameworks like Ultralytics.
    - "file_paths_only": preprocessing returns file paths/manifests, modeling loads lazily/batch-wise.
- The preprocessing_recommendations should be appropriate with the chosen model and task.
- The feature_engineering block should be actionable but still abstract (no dataset-specific code or API names). Focus on what features to create and why.
"""

    def build(
        self,
        iteration_type: str,
        task_context: Dict[str, Any],
        model_suggestions: Dict[str, Any] | None = None,
        architecture_suggestions: Dict[str, Any] | None = None,
    ) -> str:
        return self.template.format(
            iteration_type=iteration_type or "unknown",
            task_context_json=json.dumps(task_context or {}, indent=2, ensure_ascii=False),
            model_suggestions_json=json.dumps(model_suggestions or {}, indent=2, ensure_ascii=False),
            architecture_suggestions_json=json.dumps(architecture_suggestions or {}, indent=2, ensure_ascii=False),
        )

    def parse(self, response: str) -> Dict[str, Any]:
        try:
            cleaned = response.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned)
        except Exception as e:
            return {
                "error": f"Invalid JSON response from LLM: {e}",
                "raw_response": response,
            }
