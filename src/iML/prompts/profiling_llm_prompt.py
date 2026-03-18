import json
from typing import Any, Dict, List, Optional

from .base_prompt import BasePrompt


class ProfilingLLMPrompt(BasePrompt):
    """
    Prompt to generate a SAFE, LIGHTWEIGHT dataset profiling script.
    The script must:
    - read only from the provided allowed_files list
    - print ONE JSON object between markers
    - never train models, never download, never write dataset files
    """

    def default_template(self) -> str:
        # NOTE: Must escape braces because build() uses format_map.
        return """You are an ML data profiling agent.

Your task: write a SINGLE Python script that profiles a raw dataset directory D under strict constraints.
The script will be executed by the system. It MUST be safe and fast.

## INPUTS
DATASET_ROOT (absolute path):
{dataset_root}

ALLOWED_FILES (absolute paths; you MUST NOT read any other files):
```json
{allowed_files_json}
```

DESCRIPTION_ANALYSIS (JSON):
```json
{description_json}
```

DATAFILE_STRUCTURE (human-readable; may be truncated):
{datafile_structure}

BASIC_INVENTORY (JSON):
```json
{basic_inventory_json}
```

BASIC_PROFILES (JSON; sampled):
```json
{basic_profiles_json}
```

## PROFILING CONTRACT (STRICT)
- Only use the ALLOWED_FILES list; do NOT scan the filesystem beyond it.
- No network calls. No downloading. No web search. No model training.
- Do NOT write or modify any dataset files.
- Reading policy:
  - tabular (csv/tsv/parquet/xlsx/json): read <= {tabular_nrows} rows per file max.
  - text: read <= {text_max_bytes} bytes per file max.
  - images/audio: read metadata only (width/height/mode; wav sr/channels/duration).
- The script MUST print exactly one JSON object between markers:
  - Print a line: ===PROFILING_JSON_START===
  - Print the JSON object (single JSON, can be pretty-printed)
  - Print a line: ===PROFILING_JSON_END===
- Other prints are allowed BUT the markers must exist and the JSON must be valid.

## REQUIRED OUTPUT JSON SCHEMA
The JSON object MUST have these top-level keys:
{{
  "inventory": {{"counts_by_ext": object, "examples_by_ext": object}},
  "samples": [{{"path": str, "rel_path": str, "extension": str}}],
  "schemas": {{
    "tabular": [{{"rel_path": str, "columns": [str], "dtypes": object, "nrows_sampled": int}}],
    "text": [{{"rel_path": str, "kind": "tabular|text", "sniff": object}}],
    "media": [{{"rel_path": str, "kind": "image|audio", "meta": object}}]
  }},
  "signals": {{
    "candidate_train_files": [str],
    "candidate_test_files": [str],
    "candidate_submission_files": [str],
    "notes": [str]
  }}
}}

## IMPLEMENTATION NOTES
- Use stdlib where possible.
- You MAY use pandas if available. If pandas isn't available, still produce a valid JSON with best-effort signals.
- For images: try Pillow (PIL). If not available, record an error in meta and continue.
- For wav: use stdlib wave.

## OUTPUT FORMAT (CRITICAL)
- Output ONLY a Python script. No markdown fences.
"""

    def build(
        self,
        *,
        dataset_root: str,
        allowed_files: List[str],
        description_analysis: Dict[str, Any],
        datafile_structure: str,
        basic_inventory: Dict[str, Any],
        basic_profiles: List[Dict[str, Any]],
        tabular_nrows: int,
        text_max_bytes: int,
    ) -> str:
        class _SafeDict(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        values = _SafeDict(
            dataset_root=dataset_root or "",
            allowed_files_json=json.dumps(allowed_files or [], indent=2, ensure_ascii=False),
            description_json=json.dumps(description_analysis or {}, indent=2, ensure_ascii=False),
            datafile_structure=datafile_structure or "N/A",
            basic_inventory_json=json.dumps(basic_inventory or {}, indent=2, ensure_ascii=False),
            basic_profiles_json=json.dumps(basic_profiles or [], indent=2, ensure_ascii=False),
            tabular_nrows=int(tabular_nrows),
            text_max_bytes=int(text_max_bytes),
        )
        return self.template.format_map(values)

    def parse(self, response: str) -> str:
        # Extract code (support both raw and fenced, but require returning code only).
        if "```python" in response:
            return response.split("```python")[1].split("```")[0].strip()
        if "```" in response:
            return response.split("```")[1].split("```")[0].strip()
        return response.strip()

