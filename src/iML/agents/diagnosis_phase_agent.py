import json
import logging
import os
from typing import Any, Dict, Optional

from .base_agent import BaseAgent
from .utils import init_llm
from ..prompts.diagnosis_phase_prompt import DiagnosisPhasePrompt

logger = logging.getLogger(__name__)


class DiagnosisPhaseAgent(BaseAgent):
    """Generate and execute a task-adaptive diagnosis script with a fixed JSON output contract."""

    def __init__(self, config, manager, llm_config):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=self.llm_config,
            agent_name="diagnosis_phase",
            multi_turn=self.llm_config.get("multi_turn", False),
        )
        self.prompt_handler = DiagnosisPhasePrompt(
            llm_config=self.llm_config,
            manager=self.manager,
            template=getattr(self.llm_config, "template", None),
        )

    @staticmethod
    def _extract_between(text: str, start: str, end: str) -> Optional[str]:
        if not text or start not in text or end not in text:
            return None
        return text.split(start, 1)[1].split(end, 1)[0].strip()

    def _parse_diagnosis_json(self, stdout: str) -> Dict[str, Any]:
        chunk = self._extract_between(
            stdout or "",
            self.prompt_handler.DIAGNOSIS_START,
            self.prompt_handler.DIAGNOSIS_END,
        )
        if not chunk:
            return {"success": False, "error": "missing_diagnosis_markers", "stdout_head": (stdout or "")[:2000]}
        try:
            return json.loads(chunk)
        except Exception as e:
            return {"success": False, "error": f"invalid_diagnosis_json: {e}", "raw": chunk[:4000]}

    def _load_saved_diagnosis(self, diagnosis_json_path: str) -> Dict[str, Any]:
        if not diagnosis_json_path or not os.path.exists(diagnosis_json_path):
            return {}
        try:
            with open(diagnosis_json_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            logger.warning(f"DiagnosisPhaseAgent: failed to load saved diagnosis json: {e}")
        return {}

    def __call__(
        self,
        *,
        baseline_code: str,
        description_analysis: Optional[Dict[str, Any]] = None,
        profiling_summary: Optional[Dict[str, Any]] = None,
        baseline_stdout: str = "",
        latest_execution_result: Optional[Dict[str, Any]] = None,
        iteration_type: str | None = None,
    ) -> Dict[str, Any]:
        self.manager.log_agent_start("DiagnosisPhaseAgent: generating diagnosis evidence...")

        output_folder = os.path.abspath(getattr(self.manager, "output_folder", "."))
        states_dir = os.path.join(output_folder, "states")
        diagnosis_json_path = os.path.join(output_folder, "diagnosis", "diagnosis.json")

        prompt = self.prompt_handler.build(
            description_analysis=description_analysis or {},
            profiling_summary=profiling_summary or {},
            baseline_code=baseline_code or "",
            baseline_stdout=baseline_stdout or "",
            latest_execution_result=latest_execution_result or {},
            iteration_type=iteration_type,
            output_folder=output_folder,
            states_dir=states_dir,
            diagnosis_json_path=diagnosis_json_path,
        )
        self.manager.save_and_log_states(prompt, "diagnosis/prompt.txt")

        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, "diagnosis/raw_response.txt")

        diagnosis_script = self.prompt_handler.parse(response)
        self.manager.save_and_log_states(diagnosis_script, "diagnosis/generated_script.py")

        exec_result = self.manager.execute_code(
            diagnosis_script,
            "diagnosis",
            1,
            timeout_sec=int(getattr(self.config, "per_execution_timeout", 120)),
        )

        parsed = self._parse_diagnosis_json(exec_result.get("stdout", "") or "")
        if not parsed:
            parsed = {}
        if parsed.get("error"):
            saved = self._load_saved_diagnosis(diagnosis_json_path)
            if saved:
                parsed = saved

        if not isinstance(parsed, dict) or not parsed:
            parsed = {
                "success": False,
                "task_type": "unknown",
                "primary_metric": {"name": "unknown", "value": None, "higher_is_better": None},
                "fit_status": "unknown",
                "generalization_gap": None,
                "suspected_bottlenecks": [],
                "underperforming_segments": [],
                "error_patterns": ["diagnosis_generation_or_execution_failed"],
                "recommended_change_types": ["unknown"],
                "notes": ["Diagnosis phase could not produce structured evidence."],
                "evidence_refs": [],
            }

        result = {
            "exec": exec_result,
            "diagnosis": parsed,
            "diagnosis_json_path": diagnosis_json_path,
        }
        self.manager.save_and_log_states(
            json.dumps(result, ensure_ascii=False, indent=2),
            "diagnosis/diagnosis_result.json",
        )
        self.manager.log_agent_end("DiagnosisPhaseAgent: completed.")
        return result
