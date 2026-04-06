import logging
from typing import Any, Dict

from .base_agent import BaseAgent
from .utils import init_llm
from ..prompts import ResearchSelectorPrompt

logger = logging.getLogger(__name__)


class ResearchSelectorAgent(BaseAgent):
    """Select the next proposals to promote into experiment coding."""

    def __init__(self, config: Dict[str, Any], manager: Any, llm_config: Dict[str, Any]):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="research_selector",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = ResearchSelectorPrompt(manager=manager, llm_config=self.llm_config)

    def __call__(self, proposals: Dict[str, Any], proxy_scores: Dict[str, Any], top_k: int = 2) -> Dict[str, Any]:
        self.manager.log_agent_start("ResearchSelectorAgent: selecting research proposals...")
        prompt = self.prompt_handler.build(proposals=proposals, proxy_scores=proxy_scores, top_k=top_k)
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, "research/research_selector_raw_response.txt")
        parsed = self.prompt_handler.parse(response)
        self.manager.log_agent_end("ResearchSelectorAgent: proposal selection completed.")
        return parsed
