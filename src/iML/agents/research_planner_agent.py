import logging
from typing import Any, Dict

from .base_agent import BaseAgent
from .utils import init_llm
from ..prompts import ResearchPlannerPrompt

logger = logging.getLogger(__name__)


class ResearchPlannerAgent(BaseAgent):
    """Generate structured research proposals after a successful baseline."""

    def __init__(self, config: Dict[str, Any], manager: Any, llm_config: Dict[str, Any]):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="research_planner",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = ResearchPlannerPrompt(manager=manager, llm_config=self.llm_config)

    def __call__(
        self,
        baseline_summary: Dict[str, Any],
        data_contract: Dict[str, Any],
        guideline: Dict[str, Any],
        task_context: Dict[str, Any],
        iteration_type: str | None = None,
        max_proposals: int = 6,
    ) -> Dict[str, Any]:
        self.manager.log_agent_start("ResearchPlannerAgent: generating research proposals...")
        prompt = self.prompt_handler.build(
            baseline_summary=baseline_summary,
            data_contract=data_contract,
            guideline=guideline,
            task_context=task_context,
            iteration_type=iteration_type,
            max_proposals=max_proposals,
        )
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, "research/research_planner_raw_response.txt")
        parsed = self.prompt_handler.parse(response)
        self.manager.log_agent_end("ResearchPlannerAgent: research proposals completed.")
        return parsed
