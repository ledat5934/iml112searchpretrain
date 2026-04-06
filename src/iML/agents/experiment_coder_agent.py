import logging
from pathlib import Path
from typing import Any, Dict

from .base_agent import BaseAgent
from .utils import init_llm
from ..prompts import ExperimentCoderPrompt

logger = logging.getLogger(__name__)


class ExperimentCoderAgent(BaseAgent):
    """Generate a cached-data experiment script for a selected proposal."""

    def __init__(self, config: Dict[str, Any], manager: Any, llm_config: Dict[str, Any]):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="experiment_coder",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = ExperimentCoderPrompt(manager=manager, llm_config=self.llm_config)

    def __call__(
        self,
        proposal: Dict[str, Any],
        baseline_summary: Dict[str, Any],
        data_contract: Dict[str, Any],
        baseline_modeling_code: str,
        workspace_dir: str,
        iteration_type: str | None = None,
    ) -> Dict[str, Any]:
        self.manager.log_agent_start("ExperimentCoderAgent: generating experiment training script...")
        prompt = self.prompt_handler.build(
            proposal=proposal,
            baseline_summary=baseline_summary,
            data_contract=data_contract,
            baseline_modeling_code=baseline_modeling_code,
            iteration_type=iteration_type,
        )
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, "research/experiment_coder_raw_response.txt")
        code = self.prompt_handler.parse(response)

        proposal_id = proposal.get("proposal_id", "exp_primary")
        target_dir = Path(workspace_dir) / "experiments" / proposal_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "train.py"
        target.write_text(code, encoding="utf-8")
        self.manager.log_agent_end("ExperimentCoderAgent: experiment training script generated.")
        return {"status": "success", "code": code, "path": str(target)}
