import json
import logging
from typing import Any, Dict, Optional

from .base_agent import BaseAgent
from .utils import init_llm
from ..prompts.prompt_decider_prompt import PromptDeciderPrompt

logger = logging.getLogger(__name__)


class PromptDeciderAgent(BaseAgent):
    """
    LLM-based agent that decides/fills variable prompt fields for downstream templates.

    It outputs a JSON object with fields for:
    - preprocessing prompt placeholders
    - modeling prompt placeholders
    - assembler iteration context
    """

    def __init__(self, config: Dict, manager: Any, llm_config: Dict, prompt_template: Optional[str] = None):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.prompt_template = prompt_template
        self.prompt_handler = PromptDeciderPrompt(manager=manager, llm_config=llm_config, template=prompt_template)
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="prompt_decider",
            multi_turn=llm_config.get("multi_turn", False),
        )

    def __call__(
        self,
        iteration_type: str,
        guideline: Optional[Dict[str, Any]] = None,
        knowledge_pack: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.manager.log_agent_start(f"PromptDeciderAgent: building prompt fields ({iteration_type})...")

        description_analysis = getattr(self.manager, "description_analysis", {}) or {}
        guideline = guideline or getattr(self.manager, "guideline", None) or {}

        prompt = self.prompt_handler.build(
            iteration_type=iteration_type,
            description_analysis=description_analysis,
            guideline=guideline,
            knowledge_pack=knowledge_pack or {},
        )

        save_suffix = iteration_type or "default"
        self.manager.save_and_log_states(prompt, f"prompt_decider/prompt_fields_{save_suffix}_prompt.txt")

        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, f"prompt_decider/prompt_fields_{save_suffix}_raw_response.txt")

        parsed = self.prompt_handler.parse(response)
        try:
            self.manager.save_and_log_states(
                json.dumps(parsed, indent=2, ensure_ascii=False),
                f"prompt_decider/prompt_fields_{save_suffix}.json",
            )
        except Exception:
            pass

        self.manager.log_agent_end(f"PromptDeciderAgent: prompt fields completed ({iteration_type}).")
        return parsed