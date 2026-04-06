import logging
from typing import Any, Dict

from .base_agent import BaseAgent
from .utils import init_llm
from ..prompts import ProxyEstimatorPrompt

logger = logging.getLogger(__name__)


class ProxyEstimatorAgent(BaseAgent):
    """Estimate proposal quality using zero-cost or cheap proxy reasoning."""

    def __init__(self, config: Dict[str, Any], manager: Any, llm_config: Dict[str, Any]):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="proxy_estimator",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = ProxyEstimatorPrompt(manager=manager, llm_config=self.llm_config)

    def __call__(
        self,
        baseline_summary: Dict[str, Any],
        data_contract: Dict[str, Any],
        proposals: Dict[str, Any],
        iteration_type: str | None = None,
    ) -> Dict[str, Any]:
        self.manager.log_agent_start("ProxyEstimatorAgent: scoring research proposals...")
        prompt = self.prompt_handler.build(
            baseline_summary=baseline_summary,
            data_contract=data_contract,
            proposals=proposals,
            iteration_type=iteration_type,
        )
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, "research/proxy_estimator_raw_response.txt")
        parsed = self.prompt_handler.parse(response)
        self.manager.log_agent_end("ProxyEstimatorAgent: proxy scoring completed.")
        return parsed
