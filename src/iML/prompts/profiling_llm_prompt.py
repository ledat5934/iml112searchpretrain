import json
from typing import Any, Dict, List, Optional

from .base_prompt import BasePrompt


class ProfilingLLMPrompt(BasePrompt):
    """
    Prompt to generate a SAFE, LIGHTWEIGHT dataset profiling script.
    The script must:
    - read only under DATASET_ROOT with bounded work (no arbitrary internet / no writes)
    - print a human-readable profiling report between markers (optionally still valid JSON for compatibility)
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

SYSTEM_SUGGESTED_FILES (absolute paths; hints only — you decide which paths are worth opening, and you may discover others under DATASET_ROOT within the safety rules below):
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
- You MAY only read files that live under DATASET_ROOT (resolve paths; reject path traversal). Do not read outside this tree.
- You decide which files to open: use SYSTEM_SUGGESTED_FILES as starting hints, then prioritize small, high-signal files (metadata, mappings, folds, labels, submission samples). You may skip redundant bulk files (e.g. not every image) unless needed to answer a concrete question.
- Optional bounded discovery: you MAY list/walk under DATASET_ROOT with hard caps you implement in the script (e.g. max files inspected, max depth, skip files larger than a few MB except when clearly metadata). Never blindly read huge binaries fully.
- No network calls. No downloading. No web search. No model training.
- Do NOT write or modify any dataset files.
- Reading policy for anything you open:
  - tabular (csv/tsv/parquet/xlsx/json): read <= {tabular_nrows} rows per file max.
  - text: read <= {text_max_bytes} bytes per file max.
  - images: always safe to read **header/metadata** (format, size, mode, palette). Optionally, for **at most a few** representative images (cap e.g. 3–8 files, skip if file or decoded pixels would exceed a reasonable memory budget), load pixel data and report **useful low-level stats** (see below). Do not decode thousands of images.
  - audio: metadata first (sample rate, channels, bit depth, duration via frame count). Optionally read a **short initial segment** of one file (e.g. first N frames) to estimate loudness/DC offset if cheap — cap N.
- CORE METADATA PRIORITY (CRITICAL):
  - Prioritize schemas/roles from small files that define dataset contracts: folds/splits, labels/targets, id-to-filename mapping, class/species lists.
  - When several such files exist, prefer covering all small contract files over exhaustively profiling repetitive media.

## USEFUL SIGNALS TO PRIORITIZE (WHEN FEASIBLE — STAY WITHIN CAPS ABOVE)
- **Tabular**: column names, dtypes, obvious ID/label columns, row count estimate if cheap, missingness fraction on sampled rows, constant/near-constant columns, numeric ranges (min/max/mean) on a few columns, class counts for a label column if small cardinality.
- **Images** (on the small sample you decode): width/height distribution in sample, color mode (L/RGB/RGBA), **per-channel or grayscale** mean/std/min/max (pixel intensity stats), approximate dynamic range; note if images are mostly black/white or low contrast; optional: simple uniqueness hint (e.g. all pixels identical → broken file).
- **Audio**: duration, sample rate, mono/stereo, clipping hints if you peek samples; silence ratio on a short prefix if computed cheaply.
- **Cross-file / task fit**: alignment between CSV `id` and image filenames, duplicate keys, train vs test column parity, submission column names vs label space.

## WHAT TO REPORT (FREE FORM, YOUR CHOICE)
- Output a clear **prose** profiling report (plain text). Structure it however helps: sections, bullet lists, short tables in text, etc.
- Adapt depth to file type: rich but bounded — e.g. tabular schema + quick stats; for images combine **metadata for many** with **pixel-level stats for a tiny sample**; head/snippet for tiny text configs.
- Include any **interesting, task-relevant** observations (join keys, ID formats, leakage risks, class imbalance hints, duplicate IDs, etc.) when evidence exists in the sampled data.
- You may still output valid JSON between the markers instead of prose if you prefer, but there is **no** required JSON schema — content quality matters more than shape.

## OUTPUT MARKERS (CRITICAL)
The script MUST print exactly one block between these lines (prose or JSON):
- Print a line: ===PROFILING_REPORT_START===
- Print the report body (single coherent block; if JSON, it must be valid JSON)
- Print a line: ===PROFILING_REPORT_END===
Other stdout before/after is allowed, but the marked block must exist.

Legacy compatibility: ===PROFILING_JSON_START=== / ===PROFILING_JSON_END=== are also acceptable if you emit JSON.

## IMPLEMENTATION NOTES
- Use stdlib where possible.
- You MAY use pandas if available; otherwise use csv or best-effort parsing.
- For images: try Pillow (PIL); **numpy** (`numpy.asarray(image)`) is fine for mean/std on the small decoded sample. If libraries are missing, report metadata only and say so.
- For wav: use stdlib wave; for other audio formats, metadata-only unless a safe reader exists in the environment.

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

