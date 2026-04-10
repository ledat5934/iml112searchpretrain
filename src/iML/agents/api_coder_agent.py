import logging
from pathlib import Path
from typing import Any, Dict

from .base_agent import BaseAgent
from .utils import init_llm
from ..prompts import APICoderPrompt

logger = logging.getLogger(__name__)


class APICoderAgent(BaseAgent):
    """Generate FastAPI deployment wrapper for the predictor bundle."""

    def __init__(self, config: Dict[str, Any], manager: Any, llm_config: Dict[str, Any]):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="deployment_api_coder",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = APICoderPrompt(manager=manager, llm_config=self.llm_config)

    def __call__(
        self,
        description_analysis: Dict[str, Any],
        guideline: Dict[str, Any],
        predictor_code: str,
        schemas_code: str,
        workspace_dir: str,
        iteration_type: str | None = None,
    ) -> Dict[str, Any]:
        self.manager.log_agent_start("APICoderAgent: generating deployment API...")
        prompt = self.prompt_handler.build(
            description_analysis=description_analysis,
            guideline=guideline,
            predictor_code=predictor_code,
            schemas_code=schemas_code,
            iteration_type=iteration_type,
        )
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, "deployment/api_coder_raw_response.txt")
        try:
            code = self.prompt_handler.parse(response)
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}
        path = Path(workspace_dir) / "api.py"
        path.write_text(code, encoding="utf-8")
        self.manager.log_agent_end("APICoderAgent: deployment API generated.")
        return {"status": "success", "path": str(path), "code": code}
