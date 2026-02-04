# src/iML/agents/evidence_gathering_agent.py
import logging
from typing import Any, Dict

from .base_agent import BaseAgent
from ..prompts.evidence_gathering_prompt import EvidenceGatheringPrompt
from .utils import init_llm

logger = logging.getLogger(__name__)


class EvidenceGatheringAgent(BaseAgent):
    """
    Generate and run a lightweight evidence-gathering script.
    """

    def __init__(self, config, manager, llm_config, prompt_template=None):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.prompt_template = prompt_template
        self.prompt_handler = EvidenceGatheringPrompt(
            llm_config=self.llm_config,
            manager=self.manager,
            template=self.prompt_template,
        )
        self.llm = init_llm(
            llm_config=self.llm_config,
            agent_name="evidence_gathering_agent",
            multi_turn=self.llm_config.get("multi_turn", False),
        )

    def __call__(
        self,
        stderr: str,
        code_snippet: str,
        datafile_structure: str,
        task_schema: Dict[str, Any],
        phase_name: str,
        attempt: int,
    ) -> Dict[str, Any]:
        self.manager.log_agent_start("EvidenceGatheringAgent: generating evidence script...")
        prompt = self.prompt_handler.build(
            stderr=stderr,
            code_snippet=code_snippet,
            datafile_structure=datafile_structure,
            task_schema=task_schema,
        )
        self.manager.save_and_log_states(prompt, f"debug_precheck/{phase_name}/attempt_{attempt}/evidence_prompt.txt")
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, f"debug_precheck/{phase_name}/attempt_{attempt}/evidence_raw_response.txt")

        script = self.prompt_handler.parse(response)
        self.manager.save_and_log_states(script, f"debug_precheck/{phase_name}/attempt_{attempt}/evidence_script.py")

        # Execute evidence script and capture stdout/stderr
        result = self.manager.execute_code(script, f"debug_precheck/{phase_name}/attempt_{attempt}/evidence", attempt)
        self.manager.log_agent_end("EvidenceGatheringAgent: evidence collection completed.")
        return {
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "success": bool(result.get("success")),
        }
