import logging
from pathlib import Path
from typing import Any, Dict

from .base_agent import BaseAgent
from .utils import init_llm
from ..prompts import APITestPrompt

logger = logging.getLogger(__name__)


class APITestAgent(BaseAgent):
    """Generate API validation script for the deployment bundle."""

    def __init__(self, config: Dict[str, Any], manager: Any, llm_config: Dict[str, Any]):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="deployment_api_test",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = APITestPrompt(manager=manager, llm_config=self.llm_config)

    def __call__(
        self,
        description_analysis: Dict[str, Any],
        guideline: Dict[str, Any],
        predictor_code: str,
        api_code: str,
        workspace_dir: str,
        iteration_type: str | None = None,
    ) -> Dict[str, Any]:
        self.manager.log_agent_start("APITestAgent: generating deployment API validation...")
        prompt = self.prompt_handler.build(
            description_analysis=description_analysis,
            guideline=guideline,
            predictor_code=predictor_code,
            api_code=api_code,
            iteration_type=iteration_type,
        )
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, "deployment/api_test_raw_response.txt")
        try:
            code = self.prompt_handler.parse(response)
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}
        path = Path(workspace_dir) / "validate_api.py"
        path.write_text(code, encoding="utf-8")
        self.manager.log_agent_end("APITestAgent: deployment API validation generated.")
        return {"status": "success", "path": str(path), "code": code}
