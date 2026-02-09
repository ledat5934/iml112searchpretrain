import json
from typing import Dict, Any

from .base_prompt import BasePrompt


class ProfilingSummarizerPrompt(BasePrompt):
    """Prompt to condense verbose profiling into compact JSON signals for pipeline."""

    def default_template(self) -> str:
        tmpl = (
            """
You are a senior ML data analyst. Read the dataset description and the RAW profiling result below.
Your job is to produce a COMPACT, ACTIONABLE JSON summary for downstream preprocessing and modeling.

Focus on clear, minimal signals. Avoid dumping large structures. Do not include extraneous detail.

MUST OUTPUT ONLY VALID JSON with these keys:
{{
    "dataset_name": str,
    "task_type_hint": str | null,  // optional hint from description (e.g., classification, regression)
    "files": [  // short overview of key files detected from profiling summaries
        {{"name": str, "n_rows": int, "n_cols": int}}
    ],
    "label_analysis": {{
        "has_label_column": bool,
        "label_column": str | null,
        "has_missing_labels": bool | null,
        "num_classes": int | null,
        "class_distribution_imbalance": "none|mild|moderate|severe|null",
        "notes": str
    }},
    "feature_quality": {{
        "high_missing_columns": [str],
        "constant_or_near_constant_cols": [str],
        "high_cardinality_categoricals": [str],
        "date_like_cols": [str]
    }},
    "data_split_hint": {{
        "train_file": str | null,
        "test_file": str | null,
        "sample_submission_file": str | null
    }},
    "id_format_analysis": {{
        "has_file_extensions": bool,
        "detected_extensions": [str],
        "format_notes": [str]
    }},
    "domain_specific_stats": {{
        // Include ONLY sections relevant to the dataset. Omit sections with no data.
        "image": {{  // include only if image_stats is present in RAW_PROFILING
            "total_files": int,
            "avg_dimensions": str,  // e.g. "224x224"
            "uniform_dimensions": bool,
            "channels": str,  // e.g. "3-channel RGB" or "mixed (RGB + grayscale)"
            "formats": [str],  // e.g. ["JPEG", "PNG"]
            "num_classes_from_folders": int | null,
            "preprocessing_notes": str  // e.g. "Variable sizes require resizing; mostly RGB"
        }},
        "audio": {{  // include only if audio_stats is present in RAW_PROFILING
            "total_files": int,
            "duration_range": str,  // e.g. "0.5s - 30s, avg 4.2s"
            "sample_rates": str,  // e.g. "16000 Hz" or "mixed (16000, 22050)"
            "channels": str,  // e.g. "mono" or "stereo"
            "formats": [str],
            "preprocessing_notes": str  // e.g. "Variable durations require padding/truncation to fixed length"
        }},
        "text": {{  // include only if text_stats is present in RAW_PROFILING
            "text_columns": [str],  // column names containing text
            "avg_char_length": float,
            "avg_word_count": float,
            "max_word_count": int,
            "preprocessing_notes": str  // e.g. "Long texts (avg 150 words) suggest transformer tokenization with max_length=256"
        }},
        "object_detection": {{  // include only if object_detection_stats is present in RAW_PROFILING
            "annotation_format": str,  // "COCO", "YOLO", "Pascal VOC", or "unknown"
            "total_files": int,
            "total_annotations": int,
            "num_categories": int,
            "category_names": [str],  // top category names
            "avg_annotations_per_image": float,
            "class_imbalance_notes": str | null,  // e.g. "Severe imbalance: 'car' has 10x more annotations than 'bicycle'"
            "preprocessing_notes": str  // e.g. "COCO format detected; generate data.yaml for YOLO training or use pycocotools"
        }},
        "time_series": {{  // include only if time_series_stats is present in RAW_PROFILING
            "primary_datetime_col": str,
            "frequency": str,  // e.g. "daily", "hourly", "weekly"
            "time_range": str,  // e.g. "2020-01-01 to 2023-12-31"
            "has_gaps": bool,
            "gap_ratio": float | null,
            "trend_direction": str | null,  // "upward", "downward", "none"
            "seasonality_period_hint": int | null,
            "is_stationary": bool | null,
            "preprocessing_notes": str  // e.g. "Daily frequency with gaps; use temporal split; create lag features"
        }},
        "video": {{  // include only if video_stats is present in RAW_PROFILING
            "total_files": int,
            "avg_duration": str,  // e.g. "5.2s"
            "avg_resolution": str,  // e.g. "1280x720"
            "uniform_resolution": bool,
            "avg_fps": float,
            "formats": [str],
            "num_classes_from_folders": int | null,
            "preprocessing_notes": str  // e.g. "Variable durations; extract 16 frames per video; resize to 224x224"
        }}
    }}
}}

Rules:
- Use the provided profiling_result JSON's "summaries", "profiles", "id_format_analysis", and domain-specific stats ("image_stats", "audio_stats", "text_stats", "object_detection_stats", "time_series_stats", "video_stats") to infer signals succinctly.
- Pay special attention to ID format analysis for file extension information.
- For domain_specific_stats: ONLY include sections (image/audio/text/object_detection/time_series/video) for which raw profiling data actually exists. Set "domain_specific_stats" to null if no domain-specific profiling data is present (i.e., purely tabular dataset with no domain-specific stats).
- The "preprocessing_notes" field in each domain section is CRITICAL: provide actionable guidance (e.g., recommended resize dimensions, padding strategy, tokenizer max_length, temporal split strategy) derived from the actual data statistics.
- If unsure, set fields to null and explain briefly in notes.
- Do NOT output markdown fences. Output pure JSON only.

---
DESCRIPTION:
{description}

RAW_PROFILING:
```json
{profiling_compact}
```
"""
        )
        return tmpl

    def build(self, profiling_result: Dict[str, Any], description_analysis: Dict[str, Any]) -> str:
        # Compact the profiling_result before sending to LLM to reduce noise
        summaries = profiling_result.get("summaries", {})
        profiles = profiling_result.get("profiles", {})
        id_format_analysis = profiling_result.get("id_format_analysis", {})

        compact = {
            "summaries": summaries,  # already compact in agent; keys: file_stem: {n_rows,n_cols,dtypes,missing_pct,file_size_mb}
            # profiles can be large; we keep only light parts if present
            "profiles_light": {},
            "id_format_analysis": id_format_analysis,  # include ID format analysis
        }

        # take only a very small subset from profiles: for each file, variable types and n_unique if available
        for f, prof in profiles.items():
            vars_info = prof.get("variables", {}) if isinstance(prof, dict) else {}
            light_vars = {}
            for v, info in vars_info.items():
                light_vars[v] = {
                    "type": info.get("type"),
                    "n_unique": info.get("n_unique"),
                    "p_missing": info.get("p_missing"),
                }
            compact["profiles_light"][f] = {"variables": light_vars}

        # Include domain-specific profiling stats so the LLM can summarize them
        image_stats = profiling_result.get("image_stats")
        if image_stats:
            compact["image_stats"] = image_stats

        audio_stats = profiling_result.get("audio_stats")
        if audio_stats:
            compact["audio_stats"] = audio_stats

        text_stats = profiling_result.get("text_stats")
        if text_stats:
            compact["text_stats"] = text_stats

        object_detection_stats = profiling_result.get("object_detection_stats")
        if object_detection_stats:
            compact["object_detection_stats"] = object_detection_stats

        time_series_stats = profiling_result.get("time_series_stats")
        if time_series_stats:
            compact["time_series_stats"] = time_series_stats

        video_stats = profiling_result.get("video_stats")
        if video_stats:
            compact["video_stats"] = video_stats

        description = {
            "name": description_analysis.get("name"),
            "task": description_analysis.get("task"),
            "task_type": description_analysis.get("task_type"),
            "output_data": description_analysis.get("output_data"),
        }

        profiling_compact = json.dumps(compact, ensure_ascii=False)
        prompt = self.template.format(
            description=json.dumps(description, ensure_ascii=False),
            profiling_compact=profiling_compact,
        )
        self.manager.save_and_log_states(prompt, "profiling_summarizer_prompt.txt")
        return prompt

    def parse(self, response: str) -> Dict[str, Any]:
        try:
            clean = response.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(clean)
        except Exception:
            parsed = {"error": "Invalid JSON from LLM", "raw_response": response}
        return parsed
