import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from .utils import init_llm
from ..prompts.profiling_llm_prompt import ProfilingLLMPrompt
from ..utils.basic_file_profiler import BasicProfilerConfig, collect_sample_files_by_dir_extension
from ..utils.file_io import get_directory_structure

logger = logging.getLogger(__name__)


class ProfilingLLMAgent(BaseAgent):
    """
    B2 profiler: LLM generates a dataset profiling script under a strict contract,
    then we execute it and parse a single JSON object from stdout.
    """

    JSON_START = "===PROFILING_JSON_START==="
    JSON_END = "===PROFILING_JSON_END==="

    def __init__(self, config, manager, llm_config, prompt_template: Optional[str] = None):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.prompt_template = prompt_template
        self.prompt_handler = ProfilingLLMPrompt(
            llm_config=self.llm_config,
            manager=self.manager,
            template=self.prompt_template,
        )
        self.llm = init_llm(
            llm_config=self.llm_config,
            agent_name="profiling_llm",
            multi_turn=self.llm_config.get("multi_turn", False),
        )

    def _extract_json_between_markers(self, stdout: str) -> Optional[str]:
        if not stdout:
            return None
        if self.JSON_START in stdout and self.JSON_END in stdout:
            chunk = stdout.split(self.JSON_START, 1)[1].split(self.JSON_END, 1)[0]
            return chunk.strip()
        return None

    def _extract_best_json_object(self, text: str) -> Optional[str]:
        """
        Fallback extractor: find a JSON object substring in arbitrary stdout.
        """
        if not text:
            return None
        raw = text.strip()
        if raw.startswith("{") and raw.endswith("}"):
            return raw
        # Try to find the last large {...} block
        starts = [m.start() for m in re.finditer(r"\{", raw)]
        for start in reversed(starts[-50:]):  # bound work
            for end in range(len(raw) - 1, start, -1):
                if raw[end] == "}":
                    chunk = raw[start : end + 1]
                    if '"inventory"' in chunk and '"schemas"' in chunk:
                        return chunk
                    break
        return None

    def __call__(
        self,
        *,
        dataset_root: str,
        basic_inventory: Dict[str, Any],
        basic_profiles: List[Dict[str, Any]],
        attempt: int = 1,
    ) -> Dict[str, Any]:
        self.manager.log_agent_start("ProfilingLLMAgent: generating and running profiling script...")

        root = Path(dataset_root)
        # Deterministic allowed file list (1 file per dir+ext)
        cfg = BasicProfilerConfig(max_files_per_dir_ext=1)
        allowed_files, sampling_meta = collect_sample_files_by_dir_extension(
            root,
            cfg=cfg,
            skip_filenames={"description.txt"},
        )
        allowed_abs = [str(p) for p in allowed_files]

        datafile_structure = get_directory_structure(
            str(root),
            include_csv_summary=False,
            sample_rows=0,
            max_chars=2000,
        )

        prompt = self.prompt_handler.build(
            dataset_root=str(root),
            allowed_files=allowed_abs,
            description_analysis=getattr(self.manager, "description_analysis", {}) or {},
            datafile_structure=datafile_structure,
            basic_inventory=basic_inventory or {},
            basic_profiles=basic_profiles or [],
            tabular_nrows=cfg.tabular_nrows,
            text_max_bytes=cfg.text_max_bytes,
        )

        self.manager.save_and_log_states(prompt, f"profiling_llm/attempt_{attempt}/prompt.txt")
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, f"profiling_llm/attempt_{attempt}/raw_response.txt")

        script = self.prompt_handler.parse(response)
        self.manager.save_and_log_states(script, f"profiling_llm/attempt_{attempt}/generated_profiler.py")

        exec_result = self.manager.execute_code(script, "profiling_llm", attempt)
        stdout = exec_result.get("stdout", "") or ""
        stderr = exec_result.get("stderr", "") or ""

        extracted = self._extract_json_between_markers(stdout) or self._extract_best_json_object(stdout)
        parsed: Dict[str, Any]
        if extracted:
            try:
                parsed = json.loads(extracted)
            except Exception as e:
                parsed = {"error": f"invalid_json: {e}", "raw_extracted": extracted, "stderr": stderr}
        else:
            parsed = {"error": "no_json_found_in_stdout", "stdout_head": stdout[:2000], "stderr": stderr}

        # Attach meta for auditing
        parsed_meta = {
            "sampling_meta": sampling_meta,
            "n_allowed_files": len(allowed_abs),
        }
        try:
            self.manager.save_and_log_states(
                json.dumps({"result": parsed, "meta": parsed_meta}, ensure_ascii=False, indent=2),
                f"profiling_llm/attempt_{attempt}/result.json",
            )
        except Exception:
            pass

        self.manager.log_agent_end("ProfilingLLMAgent: completed.")
        return {"result": parsed, "meta": parsed_meta, "exec": {"success": bool(exec_result.get("success"))}}

