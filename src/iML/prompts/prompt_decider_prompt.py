import json
from typing import Any, Dict, Optional

from .base_prompt import BasePrompt


class PromptDeciderPrompt(BasePrompt):
    """
    Prompt handler that decides/fills variable prompt fields for downstream templates:
    - preprocessing_coder_prompt.py placeholders
    - modeling_coder_prompt.py placeholders
    - assembler_prompt.py iteration-specific context
    """

    def default_template(self) -> str:
        return """You are an expert ML engineer designing prompt fields for an AutoML coding agent.

You are given:
- Task description/context
- The chosen approach (iteration_type): traditional | custom_nn | pretrained
- The chosen model/algorithm (if available in guideline)
- The iteration-specific knowledge pack (no code)

Your job:
Return a SINGLE JSON object that fills the downstream prompt placeholders WITHOUT hard-coding specific libraries
(e.g., avoid saying "use torchvision/transformers/ultralytics" unless required by the input contract).

## INPUTS
ITERATION_TYPE: {iteration_type}

## TASK DESCRIPTION (JSON)
```json
{description_json}
```

## GUIDELINE (JSON) - optional
```json
{guideline_json}
```

## KNOWLEDGE PACK (JSON) - optional (no example code)
```json
{knowledge_pack_json}
```

## OUTPUT REQUIREMENTS
Return ONLY valid JSON (no markdown, no code fences) with this schema:
{{
  "preprocessing": {{
    "input_contract_notes_section": "Empty string or ready-to-insert section. Replaces {{input_contract_notes_section}}.",
    "model_input_data_section": "Empty string or ready-to-insert section. Replaces {{model_input_data_section}}.",
    "batch_processing_instruction": "1-3 lines. Replaces preprocessing placeholder {{batch_processing_instruction}}.",
    "data_return_format": "A single line that MUST start with '5.' describing preprocess_data() return. Replaces {{data_return_format}}.",
    "iteration_guidance": "Short guidance appended under '## ITERATION-SPECIFIC GUIDANCE' (preprocessing)."
  }},
  "modeling": {{
    "data_handling_instruction": "Ready-to-insert section describing exactly how modeling consumes preprocess_data() output. Replaces {{data_handling_instruction}}.",
    "input_contract_notes_section": "Empty string or ready-to-insert section. Replaces {{input_contract_notes_section}}.",
    "model_input_data_section": "Empty string or ready-to-insert section. Replaces {{model_input_data_section}}.",
    "iteration_guidance": "Short guidance appended under '## ITERATION-SPECIFIC GUIDANCE' (modeling)."
  }},
  "assembler": {{
    "iteration_guidance": "Short guidance appended under '## ITERATION-SPECIFIC CONTEXT' (assembler)."
  }}
}}

## EXAMPLE (FORMAT ONLY — adapt to the actual task/contracts)
{{
  "preprocessing": {{
    "input_contract_notes_section": "## INPUT CONTRACT NOTES (from Knowledge Retrieval)\\n- Fit transforms on train only; apply to val/test\\n- Do not create any dummy/synthetic data\\n",
    "model_input_data_section": "## MODEL INPUT DATA CONTRACT (from Knowledge Retrieval)\\n- dataframe\\n",
    "batch_processing_instruction": "Because the model input is dataframe, the batch processing instruction is: IMPORTANT: Load entire dataset into memory for traditional ML algorithms.",
    "data_return_format": "5. Create a function `preprocess_data()` that takes file_paths and returns (X_train, X_val, X_test, y_train, y_val, y_test) as in-memory DataFrames/arrays.",
    "iteration_guidance": "- Keep preprocessing minimal and deterministic.\\n- Use a single holdout split (no k-fold)."
  }},
  "modeling": {{
    "data_handling_instruction": "## IMPORTANT DATA HANDLING\\nCall `X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file_paths)` and train a model on X_train/y_train. Evaluate on X_val/y_val. Predict on X_test.",
    "input_contract_notes_section": "## INPUT CONTRACT NOTES (from Knowledge Retrieval)\\n- Ensure label encoding is consistent between train/val\\n",
    "model_input_data_section": "## MODEL INPUT DATA CONTRACT (from Knowledge Retrieval)\\n- dataframe\\n",
    "iteration_guidance": "- Prefer PyTorch over TensorFlow for NN/pretrained when applicable, but follow the guideline.\\n- Keep training efficient; use early stopping when possible."
  }},
  "assembler": {{
    "iteration_guidance": "- Ensure the final script writes submission.csv only after successful inference and validates it is not empty."
  }}
}}

## RULES (CRITICAL)
- Be consistent with the knowledge pack contracts:
  - If knowledge_pack.model_input_data exists, your fields MUST align with it.
  - Do not invent return types that conflict with the contract.
- Prefer PyTorch over TensorFlow when suggesting NN/pretrained handling, but do not name specific libraries unless required.
- Keep outputs concise and actionable.
"""

    def build(
        self,
        iteration_type: str,
        description_analysis: Dict[str, Any],
        guideline: Optional[Dict[str, Any]] = None,
        knowledge_pack: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.template.format(
            iteration_type=iteration_type or "unknown",
            description_json=json.dumps(description_analysis or {}, indent=2, ensure_ascii=False),
            guideline_json=json.dumps(guideline or {}, indent=2, ensure_ascii=False),
            knowledge_pack_json=json.dumps(knowledge_pack or {}, indent=2, ensure_ascii=False),
        )

    def parse(self, response: str) -> Dict[str, Any]:
        try:
            cleaned = response.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned)
        except Exception as e:
            return {"error": f"Invalid JSON response from LLM: {e}", "raw_response": response}
