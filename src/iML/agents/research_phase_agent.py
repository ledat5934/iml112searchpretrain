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
    max_iterations: int = 2
    diagnosis_enabled: bool = True
    ablation_enabled: bool = True
    ablation_n_candidates: int = 3
    ablation_proxy_time_budget_sec: int = 120
    ablation_proxy_exec_timeout_sec: int = 150
    n_candidates: int = 3
    proxy_time_budget_sec: int = 300
    proxy_exec_timeout_sec: int = 330  # give code time to soft-stop and wrap up
    top_k_full: int = 1
    termination_grace_sec: int = 2


class ResearchPhaseAgent(BaseAgent):
    """
    After baseline code is working, propose 3 different improvements,
    generate improved candidates, run timeboxed proxy eval, rank them,
    then run full eval for top-1. The winning full-run candidate can replace
    the incumbent baseline and seed the next research iteration.
    """

    DIAGNOSIS_START = "===DIAGNOSIS_SUMMARY_START==="
    DIAGNOSIS_END = "===DIAGNOSIS_SUMMARY_END==="

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

    def _parse_diagnosis_json(self, stdout: str) -> Dict[str, Any]:
        chunk = self._extract_between(stdout or "", self.DIAGNOSIS_START, self.DIAGNOSIS_END)
        if not chunk:
            return {"success": False, "error": "missing_diagnosis_markers", "stdout_head": (stdout or "")[:2000]}
        try:
            return json.loads(chunk)
        except Exception as e:
            return {"success": False, "error": f"invalid_diagnosis_json: {e}", "raw": chunk[:4000]}

    def _build_diagnosis_summary(self, stdout: str) -> Dict[str, Any]:
        diagnosis = self._parse_diagnosis_json(stdout)
        validation_metric = diagnosis.get("validation_metric") if isinstance(diagnosis, dict) else None
        train_metric = diagnosis.get("train_metric") if isinstance(diagnosis, dict) else None
        fit_status = diagnosis.get("fit_status", "unknown") if isinstance(diagnosis, dict) else "unknown"
        bottlenecks = diagnosis.get("suspected_bottlenecks", []) if isinstance(diagnosis, dict) else []
        return {
            "source": "baseline_stdout_diagnostics",
            "available": bool(isinstance(diagnosis, dict) and diagnosis.get("error") is None),
            "diagnosis": diagnosis,
            "headline_metric": validation_metric,
            "train_metric": train_metric,
            "fit_status": fit_status,
            "suspected_bottlenecks": bottlenecks if isinstance(bottlenecks, list) else [],
        }

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

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        cleaned = (response or "").strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
        return json.loads(cleaned)

    def _build_ablation_prompt(
        self,
        *,
        description_analysis: Optional[Dict[str, Any]] = None,
        profiling_summary: Optional[Dict[str, Any]] = None,
        baseline_code: str,
        baseline_stdout: str = "",
        iteration_type: str | None = None,
    ) -> str:
        cfg = self.phase_cfg
        return (
            "You are planning a low-cost ML ablation study.\n\n"
            "Goal: decide which subsystem should be improved first before proposing bigger improvements.\n"
            f"Propose exactly {int(cfg.ablation_n_candidates)} materially different ablation directions.\n\n"
            f"ITERATION_TYPE: {iteration_type or 'default'}\n\n"
            "DESCRIPTION_ANALYSIS:\n"
            f"{json.dumps(description_analysis or {}, indent=2, ensure_ascii=False)}\n\n"
            "PROFILING_SUMMARY:\n"
            f"{json.dumps(profiling_summary or {}, indent=2, ensure_ascii=False)}\n\n"
            "BASELINE_STDOUT_EXCERPT:\n"
            f"{(baseline_stdout or '')[-4000:]}\n\n"
            "BASELINE_CODE:\n"
            f"{baseline_code or ''}\n\n"
            "Rules:\n"
            "- Each ablation must isolate one subsystem or decision area.\n"
            "- Prefer low-cost changes that can be judged with a short proxy run.\n"
            "- Focus on areas like optimizer/schedule, regularization, architecture width/depth, feature handling, loss/objective, data augmentation, or validation setup.\n"
            "- Do not propose broad rewrites.\n"
            "- The proposed directions must be materially different from each other.\n\n"
            "Return valid JSON only with this schema:\n"
            "{\n"
            '  "study_goal": "short string",\n'
            '  "ablations": [\n'
            "    {\n"
            '      "proposal_id": "ablation_1",\n'
            '      "area": "short subsystem name",\n'
            '      "title": "short title",\n'
            '      "objective": "what to test",\n'
            '      "rationale": "why this area is worth testing first",\n'
            '      "changes": ["specific low-cost change 1", "specific low-cost change 2"],\n'
            '      "expected_metric": "metric name",\n'
            '      "risk_level": "low/medium/high"\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

    def _run_ablation_study(
        self,
        *,
        run_root: str,
        baseline_code: str,
        description_analysis: Optional[Dict[str, Any]] = None,
        profiling_summary: Optional[Dict[str, Any]] = None,
        baseline_stdout: str = "",
        iteration_type: str | None = None,
    ) -> Dict[str, Any]:
        cfg = self.phase_cfg
        if not cfg.ablation_enabled:
            return {"skipped": True, "reason": "ablation_disabled"}

        try:
            prompt = self._build_ablation_prompt(
                description_analysis=description_analysis,
                profiling_summary=profiling_summary,
                baseline_code=baseline_code,
                baseline_stdout=baseline_stdout,
                iteration_type=iteration_type,
            )
            self.manager.save_and_log_states(prompt, f"{run_root}/ablation/plan_prompt.txt")
            response = self.llm.assistant_chat(prompt)
            self.manager.save_and_log_states(response, f"{run_root}/ablation/plan_raw_response.txt")
            plan_payload = self._parse_json_response(response)
        except Exception as e:
            return {"skipped": False, "error": f"ablation_plan_failed: {e}"}

        leaderboard: List[Dict[str, Any]] = []
        best_row: Dict[str, Any] | None = None
        best_score = float("-inf")
        ablations = (plan_payload or {}).get("ablations", []) or []

        for idx, proposal in enumerate(ablations[: int(cfg.ablation_n_candidates)], start=1):
            proposal_id = proposal.get("proposal_id") or f"ablation_{idx}"
            mutate_prompt = self.prompt_handler.build(
                baseline_code=baseline_code,
                proxy_time_budget_sec=int(cfg.ablation_proxy_time_budget_sec),
                mode="mutate_from_proposal",
                proposal=proposal,
                description_analysis=description_analysis or {},
                profiling_summary=profiling_summary or {},
                stdout_excerpt=baseline_stdout or "",
                iteration_type=iteration_type,
            )
            self.manager.save_and_log_states(mutate_prompt, f"{run_root}/ablation/{proposal_id}/prompt.txt")
            mutate_response = self.llm.assistant_chat(mutate_prompt)
            self.manager.save_and_log_states(mutate_response, f"{run_root}/ablation/{proposal_id}/raw_response.txt")
            mutated_code = self.prompt_handler.parse(mutate_response)
            self.manager.save_and_log_states(mutated_code, f"{run_root}/ablation/{proposal_id}/mutated_code.py")

            proxy_phase_name = f"{run_root}/ablation/{proposal_id}/proxy"
            proxy_task_description = self.manager.build_debug_context(
                stderr="",
                code=mutated_code,
                phase_name=proxy_phase_name,
                attempt=1,
            )
            exec_result, final_code, debug_meta = self._run_with_optional_debug(
                code=mutated_code,
                phase_name=proxy_phase_name,
                attempt=1,
                task_description=proxy_task_description,
                timeout_sec=int(cfg.ablation_proxy_exec_timeout_sec),
                termination_grace_sec=int(cfg.termination_grace_sec),
            )
            self.manager.save_and_log_states(final_code, f"{run_root}/ablation/{proposal_id}/final_code.py")
            proxy = self._parse_proxy_json(exec_result.get("stdout", "") or "")
            score, reason = self._score_proxy(proxy)

            row = {
                "candidate": proposal_id,
                "proposal": proposal,
                "exec_success": bool(exec_result.get("success")),
                "used_debug_agent": bool(debug_meta.get("used")),
                "debug_ok": bool(debug_meta.get("ok")),
                "proxy": proxy,
                "score": score,
                "score_reason": reason,
            }
            leaderboard.append(row)
            if score > best_score:
                best_score = score
                best_row = row

        result = {
            "plan_payload": plan_payload,
            "leaderboard": leaderboard,
            "recommended_focus": best_row.get("proposal") if best_row else None,
            "best_candidate": best_row.get("candidate") if best_row else None,
        }
        self.manager.save_and_log_states(
            json.dumps(result, ensure_ascii=False, indent=2),
            f"{run_root}/ablation/ablation_result.json",
        )
        return result

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
        run_root: str,
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
        self.manager.save_and_log_states(prompt, f"{run_root}/{proposal_id}/full_prepare_prompt.txt")
        resp = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(resp, f"{run_root}/{proposal_id}/full_prepare_raw_response.txt")
        full_code = self.prompt_handler.parse(resp)
        self.manager.save_and_log_states(full_code, f"{run_root}/{proposal_id}/full_prepared_code.py")
        return full_code

    def _compare_candidate_vs_baseline(
        self,
        *,
        run_root: str,
        baseline_stdout: str,
        candidate_stdout: str,
        iteration_type: str | None = None,
    ) -> Dict[str, Any]:
        prompt = (
            "You are comparing a current baseline ML run against one new candidate run.\n"
            "Use ONLY the stdout evidence below. Prefer the run with better validation/test metrics. "
            "If the evidence is inconclusive, keep the baseline.\n\n"
            f"ITERATION_TYPE: {iteration_type or 'default'}\n\n"
            "Return ONLY valid JSON with this schema:\n"
            "{"
            "\"winner\": \"baseline\" | \"candidate\", "
            "\"confidence\": \"low\" | \"medium\" | \"high\", "
            "\"reason\": \"short string\""
            "}\n\n"
            "BASELINE STDOUT:\n"
            "<<<BASELINE>>>\n"
            f"{(baseline_stdout or '')[-4000:]}\n"
            "<<<END_BASELINE>>>\n\n"
            "CANDIDATE STDOUT:\n"
            "<<<CANDIDATE>>>\n"
            f"{(candidate_stdout or '')[-4000:]}\n"
            "<<<END_CANDIDATE>>>\n"
        )
        self.manager.save_and_log_states(prompt, f"{run_root}/stdout_compare_prompt.txt")
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, f"{run_root}/stdout_compare_raw_response.txt")
        try:
            cleaned = response.strip()
            if "```json" in cleaned:
                start = cleaned.find("```json") + 7
                end = cleaned.rfind("```")
                if end > start:
                    cleaned = cleaned[start:end].strip()
            elif "```" in cleaned:
                start = cleaned.find("```") + 3
                end = cleaned.rfind("```")
                if end > start:
                    cleaned = cleaned[start:end].strip()
            parsed = json.loads(cleaned)
        except Exception as e:
            parsed = {
                "winner": "baseline",
                "confidence": "low",
                "reason": f"stdout_compare_parse_failed: {e}",
            }
        if parsed.get("winner") not in {"baseline", "candidate"}:
            parsed["winner"] = "baseline"
        if parsed.get("confidence") not in {"low", "medium", "high"}:
            parsed["confidence"] = "low"
        if not parsed.get("reason"):
            parsed["reason"] = "invalid_or_missing_reason"
        self.manager.save_and_log_states(
            json.dumps(parsed, ensure_ascii=False, indent=2),
            f"{run_root}/stdout_compare_result.json",
        )
        return parsed

    def _run_iteration(
        self,
        *,
        run_root: str,
        baseline_code: str,
        description_analysis: Optional[Dict[str, Any]] = None,
        profiling_summary: Optional[Dict[str, Any]] = None,
        baseline_stdout: str = "",
        iteration_type: str | None = None,
        previous_directions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        cfg = self.phase_cfg
        diagnosis_summary = self._build_diagnosis_summary(baseline_stdout or "")
        self.manager.save_and_log_states(
            json.dumps(diagnosis_summary, ensure_ascii=False, indent=2),
            f"{run_root}/diagnosis_summary.json",
        )
        ablation_summary = self._run_ablation_study(
            run_root=run_root,
            baseline_code=baseline_code,
            description_analysis=description_analysis,
            profiling_summary=profiling_summary,
            baseline_stdout=baseline_stdout,
            iteration_type=iteration_type,
        )
        try:
            proposal_prompt = self.proposal_prompt.build(
                description_analysis=description_analysis or {},
                profiling_summary=profiling_summary or {},
                baseline_code=baseline_code,
                stdout_excerpt=baseline_stdout or "",
                n_candidates=int(cfg.n_candidates),
                iteration_type=iteration_type,
                previous_directions=previous_directions or [],
                ablation_summary=ablation_summary,
                diagnosis_summary=diagnosis_summary,
            )
            self.manager.save_and_log_states(proposal_prompt, f"{run_root}/proposals_prompt.txt")
            proposal_resp = self.llm.assistant_chat(proposal_prompt)
            self.manager.save_and_log_states(proposal_resp, f"{run_root}/proposals_raw_response.txt")
            proposal_payload = self.proposal_prompt.parse(proposal_resp)
        except Exception as e:
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
            self.manager.save_and_log_states(prompt, f"{run_root}/{proposal_id}/prompt.txt")
            resp = self.llm.assistant_chat(prompt)
            self.manager.save_and_log_states(resp, f"{run_root}/{proposal_id}/raw_response.txt")
            mutated = self.prompt_handler.parse(resp)
            self.manager.save_and_log_states(mutated, f"{run_root}/{proposal_id}/mutated_code.py")
            candidate_code[proposal_id] = mutated

            proxy_phase_name = f"{run_root}/{proposal_id}/proxy"
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
            f"{run_root}/leaderboard.json",
        )

        result: Dict[str, Any] = {
            "diagnosis_summary": diagnosis_summary,
            "ablation_summary": ablation_summary,
            "proposal_payload": proposal_payload,
            "leaderboard": leaderboard,
            "best_candidate": best_id,
        }

        if best_id is None:
            result["full_skipped"] = True
            result["full_reason"] = "no_best_candidate"
            return result

        try:
            winning_row = next((row for row in leaderboard if row.get("candidate") == best_id), None) or {}
            winning_proposal = winning_row.get("proposal") or {}
            best_proxy_code = candidate_code.get(best_id) or baseline_code
            best_code = self._prepare_full_run_code(
                run_root=run_root,
                proposal_id=best_id,
                proposal=winning_proposal,
                candidate_code=best_proxy_code,
                description_analysis=description_analysis or {},
                profiling_summary=profiling_summary or {},
                baseline_stdout=baseline_stdout or "",
                iteration_type=iteration_type,
            )

            full_phase_name = f"{run_root}/{best_id}/full"
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
                "proxy_code_path": f"states/{run_root}/{best_id}/mutated_code.py",
                "full_code_path": f"states/{run_root}/{best_id}/full_prepared_code.py",
                "code": best_code,
                "exec": full_exec,
                "used_debug_agent": bool(full_debug_meta.get("used")),
                "debug_ok": bool(full_debug_meta.get("ok")),
            }
            self.manager.save_and_log_states(
                json.dumps(result.get("full", {}), ensure_ascii=False, indent=2),
                f"{run_root}/full_result.json",
            )
        except Exception as e:
            result["full"] = {"candidate": best_id, "error": str(e)}

        return result

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
        current_baseline_code = baseline_code
        current_baseline_stdout = baseline_stdout or ""
        adopted_full: Optional[Dict[str, Any]] = None
        iteration_results: List[Dict[str, Any]] = []
        previous_directions: List[Dict[str, Any]] = []

        for iteration_idx in range(1, max(1, int(cfg.max_iterations)) + 1):
            run_root = f"research/iter_{iteration_idx}"
            iter_result = self._run_iteration(
                run_root=run_root,
                baseline_code=current_baseline_code,
                description_analysis=description_analysis,
                profiling_summary=profiling_summary,
                baseline_stdout=current_baseline_stdout,
                iteration_type=iteration_type,
                previous_directions=previous_directions,
            )
            iter_result["iteration_index"] = iteration_idx
            iteration_results.append(iter_result)

            for row in (iter_result.get("leaderboard") or []):
                proposal = row.get("proposal") or {}
                if not proposal:
                    continue
                previous_directions.append(
                    {
                        "iteration_index": iteration_idx,
                        "proposal_id": row.get("candidate"),
                        "title": proposal.get("title"),
                        "objective": proposal.get("objective"),
                        "changes": proposal.get("changes", []),
                        "score_reason": row.get("score_reason"),
                    }
                )

            full_result = (iter_result or {}).get("full", {})
            full_exec = full_result.get("exec", {}) if isinstance(full_result, dict) else {}
            if not (isinstance(full_exec, dict) and full_exec.get("success")):
                iter_result["selection"] = {
                    "winner": "baseline",
                    "confidence": "low",
                    "reason": "full_run_failed_or_missing",
                }
                continue

            compare_result = self._compare_candidate_vs_baseline(
                run_root=run_root,
                baseline_stdout=current_baseline_stdout,
                candidate_stdout=full_exec.get("stdout", "") or "",
                iteration_type=iteration_type,
            )
            iter_result["selection"] = compare_result

            if compare_result.get("winner") == "candidate":
                current_baseline_code = full_result.get("code") or current_baseline_code
                current_baseline_stdout = full_exec.get("stdout", "") or current_baseline_stdout
                adopted_full = full_result

        result: Dict[str, Any] = {
            "iterations": iteration_results,
            "final_baseline_code": current_baseline_code,
            "final_baseline_stdout": current_baseline_stdout,
        }
        if adopted_full is not None:
            result["full"] = adopted_full
        elif iteration_results:
            result["full_skipped"] = True
            result["full_reason"] = (iteration_results[-1].get("selection") or {}).get("reason", "baseline_kept")
        else:
            result["full_skipped"] = True
            result["full_reason"] = "no_iterations_ran"

        self.manager.save_and_log_states(
            json.dumps(result, ensure_ascii=False, indent=2),
            "research/research_summary.json",
        )
        self.manager.log_agent_end("ResearchPhaseAgent: completed.")
        return result

