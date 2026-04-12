import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from .utils import init_llm
from ..prompts.research_proposal_prompt import ResearchProposalPrompt
from ..prompts.research_mutation_prompt import ResearchMutationPrompt

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

            exec_result = self.manager.execute_code(
                mutated,
                phase_name=f"research/{proposal_id}/proxy",
                attempt=1,
                timeout_sec=int(cfg.proxy_exec_timeout_sec),
                termination_grace_sec=int(cfg.termination_grace_sec),
            )
            stdout = exec_result.get("stdout", "") or ""
            proxy = self._parse_proxy_json(stdout)
            score, reason = self._score_proxy(proxy)

            row = {
                "candidate": proposal_id,
                "proposal": proposal,
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
            best_code = candidate_code.get(best_id) or baseline_code

            full_exec = self.manager.execute_code(
                best_code,
                phase_name=f"research/{best_id}/full",
                attempt=1,
            )
            result["full"] = {"candidate": best_id, "code": best_code, "exec": full_exec}
            self.manager.save_and_log_states(
                json.dumps(result.get("full", {}), ensure_ascii=False, indent=2),
                "research/full_result.json",
            )
        except Exception as e:
            result["full"] = {"candidate": best_id, "error": str(e)}

        self.manager.log_agent_end("ResearchPhaseAgent: completed.")
        return result

