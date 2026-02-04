# src/iML/agents/error_triage_agent.py
import logging
from typing import Any, Dict

from .base_agent import BaseAgent
from ..prompts.error_triage_prompt import ErrorTriagePrompt
from .utils import init_llm

logger = logging.getLogger(__name__)


class ErrorTriageAgent(BaseAgent):
    """
    Decide whether to gather evidence or debug directly.
    """

    def __init__(self, config, manager, llm_config, prompt_template=None):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.prompt_template = prompt_template
        self.prompt_handler = ErrorTriagePrompt(
            llm_config=self.llm_config,
            manager=self.manager,
            template=self.prompt_template,
        )
        self.llm = init_llm(
            llm_config=self.llm_config,
            agent_name="error_triage_agent",
            multi_turn=self.llm_config.get("multi_turn", False),
        )

    def __call__(
        self,
        stderr: str,
        code_snippet: str,
        datafile_structure: str,
        task_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.manager.log_agent_start("ErrorTriageAgent: triaging error...")
        prompt = self.prompt_handler.build(
            stderr=stderr,
            code_snippet=code_snippet,
            datafile_structure=datafile_structure,
            task_schema=task_schema,
        )
        self.manager.save_and_log_states(prompt, "debug_precheck/triage_prompt.txt", add_uuid=True)
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, "debug_precheck/triage_raw_response.txt", add_uuid=True)
        triage = self.prompt_handler.parse(response)
        self.manager.log_agent_end("ErrorTriageAgent: triage completed.")
        return triage
