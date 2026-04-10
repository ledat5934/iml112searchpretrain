import logging
from pathlib import Path
from typing import Any, Dict

from .base_agent import BaseAgent
from .utils import init_llm
from ..prompts import DeploymentRefactorPrompt

logger = logging.getLogger(__name__)


class DeploymentRefactorAgent(BaseAgent):
    """Generate a deployment-ready predictor bundle from assembled code."""

    def __init__(self, config: Dict[str, Any], manager: Any, llm_config: Dict[str, Any]):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="deployment_refactor",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = DeploymentRefactorPrompt(manager=manager, llm_config=self.llm_config)

    def __call__(
        self,
        description_analysis: Dict[str, Any],
        guideline: Dict[str, Any],
        task_schema: Dict[str, Any],
        preprocessing_code: str,
        modeling_code: str,
        assembled_code: str,
        workspace_dir: str,
        iteration_type: str | None = None,
    ) -> Dict[str, Any]:
        self.manager.log_agent_start("DeploymentRefactorAgent: generating deployment bundle...")
        prompt = self.prompt_handler.build(
            description_analysis=description_analysis,
            guideline=guideline,
            task_schema=task_schema,
            preprocessing_code=preprocessing_code,
            modeling_code=modeling_code,
            assembled_code=assembled_code,
            iteration_type=iteration_type,
        )
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, "deployment/deployment_refactor_raw_response.txt")
        try:
            files = self.prompt_handler.parse(response)
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}
        written = {}
        for name, content in files.items():
            path = Path(workspace_dir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            written[name] = str(path)
        self.manager.log_agent_end("DeploymentRefactorAgent: deployment bundle generated.")
        return {"status": "success", "files": written}
