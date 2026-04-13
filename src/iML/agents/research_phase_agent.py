import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from .utils import init_llm
from ..prompts.research_proposal_prompt import ResearchProposalPrompt
from ..prompts.research_mutation_prompt import ResearchMutationPrompt
from ..utils.file_io import get_directory_structure

logger = logging.getLogger(__name__)


@dataclass
class ResearchPhaseConfig:
    enabled: bool = False
    n_candidates: int = 3
    proxy_time_budget_sec: int = 300
    proxy_exec_timeout_sec: int = 330  # give code time to soft-stop and wrap up
    top_k_full: int = 1
    termination_grace_sec: int = 2


class ResearchPhaseAgent(BaseAgent):
    """
    After baseline code is working, propose 3 different improvements,
    generate improved candidates, run timeboxed proxy eval, rank them,
    then run full eval for top-1.
    """

    def __init__(self, config, manager, llm_config, phase_cfg: Optional[ResearchPhaseConfig] = None):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.phase_cfg = phase_cfg or ResearchPhaseConfig()
        self.llm = init_llm(
            llm_config=self.llm_config,
            agent_name="research_phase",
            multi_turn=self.llm_config.get("multi_turn", False),
        )
        self.prompt_handler = ResearchMutationPrompt(
            llm_config=self.llm_config,
            manager=self.manager,
            template=getattr(self.llm_config, "template", None),
        )
        self.proposal_prompt = ResearchProposalPrompt(
            llm_config=self.llm_config,
            manager=self.manager,
            template=None,
        )

    @staticmethod
    def _extract_between(text: str, start: str, end: str) -> Optional[str]:
        if not text or start not in text or end not in text:
            return None
        return text.split(start, 1)[1].split(end, 1)[0].strip()

    def _parse_proxy_json(self, stdout: str) -> Dict[str, Any]:
        start = self.prompt_handler.PROXY_START
        end = self.prompt_handler.PROXY_END
        chunk = self._extract_between(stdout or "", start, end)
        if not chunk:
            return {"success": False, "error": "missing_proxy_markers", "stdout_head": (stdout or "")[:2000]}
        try:
            return json.loads(chunk)
        except Exception as e:
            return {"success": False, "error": f"invalid_proxy_json: {e}", "raw": chunk[:4000]}

    def _score_proxy(self, proxy: Dict[str, Any]) -> Tuple[float, str]:
        """
        Return (score, reason). Higher score is better.
        - Prefer proxy_metric if present; otherwise use negative train loss.
        """
        pm = proxy.get("proxy_metric") if isinstance(proxy, dict) else None
        if isinstance(pm, dict):
            val = pm.get("value", None)
            hib = pm.get("higher_is_better", None)
            name = pm.get("name", "proxy_metric")
            if isinstance(val, (int, float)):
                if hib is True:
                    return float(val), f"use_metric:{name}:higher_is_better"
                if hib is False:
                    return -float(val), f"use_metric:{name}:lower_is_better"
                # unknown direction: assume lower is better for common losses
                if str(name).lower() in {"logloss", "loss", "rmse", "mae", "mse"}:
                    return -float(val), f"use_metric:{name}:assume_lower_better"
                return float(val), f"use_metric:{name}:assume_higher_better"

        tl = proxy.get("fallback_train_loss", None) if isinstance(proxy, dict) else None
        if isinstance(tl, (int, float)):
            return -float(tl), "fallback_train_loss"

        return float("-inf"), "no_usable_metric"

    def _run_with_optional_debug(
        self,
        *,
        code: str,
        phase_name: str,
        attempt: int,
        task_description: str,
        timeout_sec: Optional[int] = None,
        termination_grace_sec: Optional[int] = None,
        require_submission: bool = False,
        submission_filename: str = "submission.csv",
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        exec_kwargs: Dict[str, Any] = {}
        if timeout_sec is not None:
            exec_kwargs["timeout_sec"] = int(timeout_sec)
        if termination_grace_sec is not None:
            exec_kwargs["termination_grace_sec"] = int(termination_grace_sec)

        exec_result = self.manager.execute_code(
            code,
            phase_name=phase_name,
            attempt=attempt,
            **exec_kwargs,
        )
        patched_code = code
        debug_meta: Dict[str, Any] = {"used": False}

        submission_path = os.path.join(self.manager.output_folder, submission_filename)
        needs_debug = not bool(exec_result.get("success"))
        if require_submission and not needs_debug and not os.path.exists(submission_path):
            needs_debug = True
            exec_result = {
                **exec_result,
                "success": False,
                "stderr": (
                    (exec_result.get("stderr", "") or "")
                    + f"\nExpected output artifact missing: {submission_path}\n"
                ).strip(),
            }

        if not needs_debug or not self.manager.is_debug_enabled():
            return exec_result, patched_code, debug_meta

        filename = "code_generated"
        datafile_structure = get_directory_structure(self.manager.input_data_folder)
        ok, patched_code, meta = self.manager.debug_agent.llm_debug_fix(
            code=code,
            stderr=exec_result.get("stderr", "") or "",
            phase_name=phase_name,
            filename=filename,
            attempt=attempt,
            task_description=task_description,
            datafile_structure=datafile_structure,
            require_submission=require_submission,
            submission_filename=submission_filename,
        )
        debug_meta = {
            "used": True,
            "ok": bool(ok),
            "meta": meta,
        }
        if ok:
            exec_result = meta.get("last_result", {}) or exec_result
        return exec_result, patched_code, debug_meta

    def _prepare_full_run_code(
        self,
        *,
        proposal_id: str,
        proposal: Dict[str, Any],
        candidate_code: str,
        description_analysis: Optional[Dict[str, Any]] = None,
        profiling_summary: Optional[Dict[str, Any]] = None,
        baseline_stdout: str = "",
        iteration_type: str | None = None,
    ) -> str:
        prompt = self.prompt_handler.build_full_run(
            candidate_code=candidate_code,
            proxy_time_budget_sec=int(self.phase_cfg.proxy_time_budget_sec),
            proposal=proposal,
            description_analysis=description_analysis or {},
            profiling_summary=profiling_summary or {},
            stdout_excerpt=baseline_stdout or "",
            iteration_type=iteration_type,
        )
        self.manager.save_and_log_states(prompt, f"research/{proposal_id}/full_prepare_prompt.txt")
        resp = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(resp, f"research/{proposal_id}/full_prepare_raw_response.txt")
        full_code = self.prompt_handler.parse(resp)
        self.manager.save_and_log_states(full_code, f"research/{proposal_id}/full_prepared_code.py")
        return full_code

    def __call__(
        self,
        *,
        baseline_code: str,
        description_analysis: Optional[Dict[str, Any]] = None,
        profiling_summary: Optional[Dict[str, Any]] = None,
        baseline_stdout: str = "",
        iteration_type: str | None = None,
    ) -> Dict[str, Any]:
        cfg = self.phase_cfg
        if not cfg.enabled:
            return {"skipped": True, "reason": "research_phase_disabled"}

        self.manager.log_agent_start("ResearchPhaseAgent: planning and evaluating improvements...")

        try:
            proposal_prompt = self.proposal_prompt.build(
                description_analysis=description_analysis or {},
                profiling_summary=profiling_summary or {},
                baseline_code=baseline_code,
                stdout_excerpt=baseline_stdout or "",
                n_candidates=int(cfg.n_candidates),
                iteration_type=iteration_type,
            )
            self.manager.save_and_log_states(proposal_prompt, "research/proposals_prompt.txt")
            proposal_resp = self.llm.assistant_chat(proposal_prompt)
            self.manager.save_and_log_states(proposal_resp, "research/proposals_raw_response.txt")
            proposal_payload = self.proposal_prompt.parse(proposal_resp)
        except Exception as e:
            self.manager.log_agent_end("ResearchPhaseAgent: proposal generation failed.")
            return {"skipped": False, "error": f"proposal_generation_failed: {e}"}

        leaderboard: List[Dict[str, Any]] = []
        candidate_code: Dict[str, str] = {}
        best_id: Optional[str] = None
        best_score = float("-inf")
        proposals = (proposal_payload or {}).get("proposals", []) or []

        for i, proposal in enumerate(proposals[: int(cfg.n_candidates)], start=1):
            proposal_id = proposal.get("proposal_id") or f"candidate_{i}"
            prompt = self.prompt_handler.build(
                baseline_code=baseline_code,
                proxy_time_budget_sec=int(cfg.proxy_time_budget_sec),
                mode="mutate_from_proposal",
                proposal=proposal,
                description_analysis=description_analysis or {},
                profiling_summary=profiling_summary or {},
                stdout_excerpt=baseline_stdout or "",
                iteration_type=iteration_type,
            )
            self.manager.save_and_log_states(prompt, f"research/{proposal_id}/prompt.txt")
            resp = self.llm.assistant_chat(prompt)
            self.manager.save_and_log_states(resp, f"research/{proposal_id}/raw_response.txt")
            mutated = self.prompt_handler.parse(resp)
            self.manager.save_and_log_states(mutated, f"research/{proposal_id}/mutated_code.py")
            candidate_code[proposal_id] = mutated

            proxy_phase_name = f"research/{proposal_id}/proxy"
            proxy_task_description = self.manager.build_debug_context(
                stderr="",
                code=mutated,
                phase_name=proxy_phase_name,
                attempt=1,
            )
            exec_result, final_code, debug_meta = self._run_with_optional_debug(
                code=mutated,
                phase_name=proxy_phase_name,
                attempt=1,
                task_description=proxy_task_description,
                timeout_sec=int(cfg.proxy_exec_timeout_sec),
                termination_grace_sec=int(cfg.termination_grace_sec),
            )
            candidate_code[proposal_id] = final_code
            stdout = exec_result.get("stdout", "") or ""
            proxy = self._parse_proxy_json(stdout)
            score, reason = self._score_proxy(proxy)

            row = {
                "candidate": proposal_id,
                "proposal": proposal,
                "used_debug_agent": bool(debug_meta.get("used")),
                "debug_ok": bool(debug_meta.get("ok")),
                "exec_success": bool(exec_result.get("success")),
                "proxy": proxy,
                "score": score,
                "score_reason": reason,
            }
            leaderboard.append(row)

            if score > best_score:
                best_score = score
                best_id = proposal_id

        # Persist leaderboard
        self.manager.save_and_log_states(
            json.dumps(
                {
                    "proposal_payload": proposal_payload,
                    "leaderboard": leaderboard,
                    "best_candidate": best_id,
                },
                ensure_ascii=False,
                indent=2,
            ),
            "research/leaderboard.json",
        )

        result: Dict[str, Any] = {
            "proposal_payload": proposal_payload,
            "leaderboard": leaderboard,
            "best_candidate": best_id,
        }

        if best_id is None:
            self.manager.log_agent_end("ResearchPhaseAgent: no usable proxy score; skipping full.")
            result["full_skipped"] = True
            result["full_reason"] = "no_best_candidate"
            return result

        # Full run for top-1 candidate
        try:
            winning_row = next((row for row in leaderboard if row.get("candidate") == best_id), None) or {}
            winning_proposal = winning_row.get("proposal") or {}
            best_proxy_code = candidate_code.get(best_id) or baseline_code
            best_code = self._prepare_full_run_code(
                proposal_id=best_id,
                proposal=winning_proposal,
                candidate_code=best_proxy_code,
                description_analysis=description_analysis or {},
                profiling_summary=profiling_summary or {},
                baseline_stdout=baseline_stdout or "",
                iteration_type=iteration_type,
            )

            full_phase_name = f"research/{best_id}/full"
            full_task_description = self.manager.build_debug_context(
                stderr="",
                code=best_code,
                phase_name=full_phase_name,
                attempt=1,
            )
            full_exec, best_code, full_debug_meta = self._run_with_optional_debug(
                code=best_code,
                phase_name=full_phase_name,
                attempt=1,
                task_description=full_task_description,
                require_submission=True,
                submission_filename="submission.csv",
            )
            result["full"] = {
                "candidate": best_id,
                "proposal": winning_proposal,
                "proxy_code_path": f"states/research/{best_id}/mutated_code.py",
                "full_code_path": f"states/research/{best_id}/full_prepared_code.py",
                "code": best_code,
                "exec": full_exec,
                "used_debug_agent": bool(full_debug_meta.get("used")),
                "debug_ok": bool(full_debug_meta.get("ok")),
            }
            self.manager.save_and_log_states(
                json.dumps(result.get("full", {}), ensure_ascii=False, indent=2),
                "research/full_result.json",
            )
        except Exception as e:
            result["full"] = {"candidate": best_id, "error": str(e)}

        self.manager.log_agent_end("ResearchPhaseAgent: completed.")
        return result

