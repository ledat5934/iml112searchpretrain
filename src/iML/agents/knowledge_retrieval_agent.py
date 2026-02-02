# src/iML/agents/knowledge_retrieval_agent.py
import json
import logging
from typing import Any, Dict

from .base_agent import BaseAgent
from ..prompts.knowledge_retrieval_prompt import KnowledgeRetrievalPrompt
from .utils import init_llm

logger = logging.getLogger(__name__)


class KnowledgeRetrievalAgent(BaseAgent):
    """
    LLM-based agent that produces iteration-specific knowledge packs.
    No example code is allowed in the output.
    """

    def __init__(self, config, manager, llm_config, prompt_template=None):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.prompt_template = prompt_template
        self.prompt_handler = KnowledgeRetrievalPrompt(
            llm_config=self.llm_config,
            manager=self.manager,
            template=self.prompt_template,
        )
        self.llm = init_llm(
            llm_config=self.llm_config,
            agent_name="knowledge_retrieval_agent",
            multi_turn=self.llm_config.get("multi_turn", False),
        )

    def __call__(self, iteration_type: str, model_suggestions: Dict[str, Any] | None = None, architecture_suggestions: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.manager.log_agent_start(f"KnowledgeRetrievalAgent: building knowledge pack ({iteration_type})...")

        task_context = getattr(self.manager, "task_context", {}) or {}
        if not task_context:
            logger.warning("KnowledgeRetrievalAgent: task_context missing; proceeding with empty context.")

        prompt = self.prompt_handler.build(
            iteration_type=iteration_type,
            task_context=task_context,
            model_suggestions=model_suggestions,
            architecture_suggestions=architecture_suggestions,
        )
        save_suffix = iteration_type or "default"
        self.manager.save_and_log_states(prompt, f"knowledge/knowledge_{save_suffix}_prompt.txt")

        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, f"knowledge/knowledge_{save_suffix}_raw_response.txt")

        knowledge_pack = self.prompt_handler.parse(response)
        try:
            self.manager.save_and_log_states(
                json.dumps(knowledge_pack, indent=2, ensure_ascii=False),
                f"knowledge/knowledge_{save_suffix}.json",
            )
        except Exception:
            pass

        self.manager.log_agent_end(f"KnowledgeRetrievalAgent: knowledge pack completed ({iteration_type}).")
        return knowledge_pack
