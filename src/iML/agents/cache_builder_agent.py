import logging
from pathlib import Path
from typing import Any, Dict

from .base_agent import BaseAgent
from .utils import init_llm
from ..prompts import CacheBuilderPrompt

logger = logging.getLogger(__name__)


class CacheBuilderAgent(BaseAgent):
    """Generate a cache builder script for the post-baseline research workspace."""

    def __init__(self, config: Dict[str, Any], manager: Any, llm_config: Dict[str, Any]):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="cache_builder",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = CacheBuilderPrompt(manager=manager, llm_config=self.llm_config)

    def __call__(
        self,
        description_analysis: Dict[str, Any],
        data_contract: Dict[str, Any],
        preprocessing_code: str,
        workspace_dir: str,
        iteration_type: str | None = None,
    ) -> Dict[str, Any]:
        self.manager.log_agent_start("CacheBuilderAgent: generating cache builder script...")

        if not preprocessing_code:
            return {"status": "failed", "error": "preprocessing code not available"}

        prompt = self.prompt_handler.build(
            description_analysis=description_analysis,
            data_contract=data_contract,
            preprocessing_code=preprocessing_code,
            iteration_type=iteration_type,
        )
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, "research/cache_builder_raw_response.txt")
        code = self.prompt_handler.parse(response)

        target = Path(workspace_dir) / "cache_builder.py"
        target.write_text(code, encoding="utf-8")
        self.manager.log_agent_end("CacheBuilderAgent: cache builder script generated.")
        return {"status": "success", "code": code, "path": str(target)}
