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

    DEFAULT_MAX_ROUNDS = 3

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

    @staticmethod
    def _default_diagnosis() -> Dict[str, Any]:
        return {
            "success": False,
            "task_type": "unknown",
            "primary_metric": {"name": "unknown", "value": None, "higher_is_better": None},
            "fit_status": "unknown",
            "generalization_gap": None,
            "confidence": "low",
            "top_bottleneck": {"area": "unknown", "reason": "No structured diagnosis was produced.", "evidence": []},
            "suspected_bottlenecks": [],
            "underperforming_segments": [],
            "error_patterns": ["diagnosis_generation_or_execution_failed"],
            "recommended_change_types": ["unknown"],
            "artifact_paths": {
                "model": [],
                "preprocessor": [],
                "validation_predictions": [],
                "training_history": [],
                "metadata": [],
                "other": [],
            },
            "debug_actions_taken": [],
            "next_round_focus": [],
            "needs_another_round": False,
            "stop_diagnosis": True,
            "notes": ["Diagnosis phase could not produce structured evidence."],
            "evidence_refs": [],
        }

    def _normalize_diagnosis(self, parsed: Dict[str, Any], *, diagnosis_json_path: str, round_index: int) -> Dict[str, Any]:
        diagnosis = self._default_diagnosis()
        if isinstance(parsed, dict):
            diagnosis.update(parsed)

        if diagnosis.get("confidence") not in {"low", "medium", "high"}:
            diagnosis["confidence"] = "low"
        if diagnosis.get("fit_status") not in {"underfit", "overfit", "balanced", "unknown"}:
            diagnosis["fit_status"] = "unknown"

        metric = diagnosis.get("primary_metric")
        if not isinstance(metric, dict):
            diagnosis["primary_metric"] = {"name": "unknown", "value": None, "higher_is_better": None}

        top_bottleneck = diagnosis.get("top_bottleneck")
        if not isinstance(top_bottleneck, dict):
            top_bottleneck = {}
        diagnosis["top_bottleneck"] = {
            "area": top_bottleneck.get("area", "unknown"),
            "reason": top_bottleneck.get("reason", "No bottleneck reason provided."),
            "evidence": top_bottleneck.get("evidence", []) if isinstance(top_bottleneck.get("evidence"), list) else [],
        }

        for key in (
            "suspected_bottlenecks",
            "underperforming_segments",
            "error_patterns",
            "recommended_change_types",
            "debug_actions_taken",
            "next_round_focus",
            "notes",
            "evidence_refs",
        ):
            if not isinstance(diagnosis.get(key), list):
                diagnosis[key] = []

        artifact_paths = diagnosis.get("artifact_paths")
        if not isinstance(artifact_paths, dict):
            artifact_paths = {}
        diagnosis["artifact_paths"] = {
            "model": artifact_paths.get("model", []) if isinstance(artifact_paths.get("model"), list) else [],
            "preprocessor": artifact_paths.get("preprocessor", []) if isinstance(artifact_paths.get("preprocessor"), list) else [],
            "validation_predictions": artifact_paths.get("validation_predictions", []) if isinstance(artifact_paths.get("validation_predictions"), list) else [],
            "training_history": artifact_paths.get("training_history", []) if isinstance(artifact_paths.get("training_history"), list) else [],
            "metadata": artifact_paths.get("metadata", []) if isinstance(artifact_paths.get("metadata"), list) else [],
            "other": artifact_paths.get("other", []) if isinstance(artifact_paths.get("other"), list) else [],
        }
        if diagnosis_json_path and diagnosis_json_path not in diagnosis["artifact_paths"]["metadata"]:
            diagnosis["artifact_paths"]["metadata"].append(diagnosis_json_path)

        has_top_bottleneck = diagnosis["top_bottleneck"]["area"] not in {"", "unknown"} and bool(
            diagnosis["top_bottleneck"]["reason"]
        )
        explicit_stop = bool(diagnosis.get("stop_diagnosis"))
        needs_another_round = bool(diagnosis.get("needs_another_round"))
        if explicit_stop:
            diagnosis["needs_another_round"] = False
        elif diagnosis["confidence"] in {"medium", "high"} and has_top_bottleneck:
            diagnosis["stop_diagnosis"] = True
            diagnosis["needs_another_round"] = False
        else:
            diagnosis["stop_diagnosis"] = False
            diagnosis["needs_another_round"] = needs_another_round or not has_top_bottleneck
            if not diagnosis["next_round_focus"] and not has_top_bottleneck:
                diagnosis["next_round_focus"] = ["Collect stronger evidence for the most likely bottleneck."]

        diagnosis["diagnosis_round"] = int(round_index)
        return diagnosis

    def _should_stop(self, diagnosis: Dict[str, Any], round_index: int, max_rounds: int) -> bool:
        if round_index >= max_rounds:
            return True
        return bool(diagnosis.get("stop_diagnosis")) or not bool(diagnosis.get("needs_another_round"))

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
        artifacts_dir = os.path.join(output_folder, "artifacts")
        diagnosis_json_path = os.path.join(output_folder, "diagnosis", "diagnosis.json")
        research_phase_cfg = getattr(self.config, "research_phase", None)
        if isinstance(research_phase_cfg, dict):
            max_rounds = int(research_phase_cfg.get("diagnosis_max_rounds", self.DEFAULT_MAX_ROUNDS))
        else:
            max_rounds = int(getattr(research_phase_cfg, "diagnosis_max_rounds", self.DEFAULT_MAX_ROUNDS))
        if max_rounds < 1:
            max_rounds = self.DEFAULT_MAX_ROUNDS

        previous_diagnosis: Dict[str, Any] = {}
        latest_exec_result: Dict[str, Any] = {}
        round_results: list[Dict[str, Any]] = []
        final_diagnosis = self._default_diagnosis()

        for round_index in range(1, max_rounds + 1):
            round_root = f"diagnosis/round_{round_index}"
            prompt = self.prompt_handler.build(
                description_analysis=description_analysis or {},
                profiling_summary=profiling_summary or {},
                baseline_code=baseline_code or "",
                baseline_stdout=baseline_stdout or "",
                latest_execution_result=latest_execution_result or {},
                previous_diagnosis=previous_diagnosis,
                round_index=round_index,
                max_rounds=max_rounds,
                iteration_type=iteration_type,
                output_folder=output_folder,
                states_dir=states_dir,
                artifacts_dir=artifacts_dir,
                diagnosis_json_path=diagnosis_json_path,
            )
            self.manager.save_and_log_states(prompt, "diagnosis/prompt.txt")
            self.manager.save_and_log_states(prompt, f"{round_root}/prompt.txt")

            response = self.llm.assistant_chat(prompt)
            self.manager.save_and_log_states(response, "diagnosis/raw_response.txt")
            self.manager.save_and_log_states(response, f"{round_root}/raw_response.txt")

            diagnosis_script = self.prompt_handler.parse(response)
            self.manager.save_and_log_states(diagnosis_script, "diagnosis/generated_script.py")
            self.manager.save_and_log_states(diagnosis_script, f"{round_root}/generated_script.py")

            exec_result = self.manager.execute_code(
                diagnosis_script,
                "diagnosis",
                round_index,
                timeout_sec=int(getattr(self.config, "per_execution_timeout", 120)),
            )
            latest_exec_result = exec_result

            stdout = exec_result.get("stdout", "") or ""
            self.manager.save_and_log_states(stdout, f"{round_root}/stdout.txt")
            self.manager.save_and_log_states(exec_result.get("stderr", "") or "", f"{round_root}/stderr.txt")

            parsed = self._parse_diagnosis_json(stdout)
            if not parsed:
                parsed = {}
            if parsed.get("error"):
                saved = self._load_saved_diagnosis(diagnosis_json_path)
                if saved:
                    parsed = saved

            normalized = self._normalize_diagnosis(
                parsed if isinstance(parsed, dict) else {},
                diagnosis_json_path=diagnosis_json_path,
                round_index=round_index,
            )
            previous_diagnosis = normalized
            final_diagnosis = normalized

            round_result = {
                "round_index": round_index,
                "exec": exec_result,
                "diagnosis": normalized,
            }
            round_results.append(round_result)
            self.manager.save_and_log_states(
                json.dumps(round_result, ensure_ascii=False, indent=2),
                f"{round_root}/round_result.json",
            )

            if self._should_stop(normalized, round_index, max_rounds):
                break

        result = {
            "exec": latest_exec_result,
            "diagnosis": final_diagnosis,
            "diagnosis_json_path": diagnosis_json_path,
            "rounds": round_results,
        }
        self.manager.save_and_log_states(
            json.dumps(result, ensure_ascii=False, indent=2),
            "diagnosis/diagnosis_result.json",
        )
        self.manager.log_agent_end("DiagnosisPhaseAgent: completed.")
        return result
