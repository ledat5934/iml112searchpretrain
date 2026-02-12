# src/iML/agents/modeling_coder_agent.py
import logging
from typing import Dict, Any

from .base_agent import BaseAgent
from ..prompts import ModelingCoderPrompt
from .utils import init_llm
from ..utils.file_io import get_directory_structure

logger = logging.getLogger(__name__)

class ModelingCoderAgent(BaseAgent):
    """
    Agent to create modeling code.
    It only generates code once and does not execute it.
    """
    def __init__(self, config: Dict, manager: Any, llm_config: Dict, **kwargs):
        super().__init__(config, manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="modeling_coder",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = ModelingCoderPrompt(
            manager=manager, 
            llm_config=self.llm_config
        )

    def __call__(self, iteration_type=None) -> Dict[str, Any]:
        """
        Agent for generating code based on training requirements.
        """
        self.manager.log_agent_start("Starting modeling code generation...")

        guideline = self.manager.guideline
        description = self.manager.description_analysis
        preprocessing_code = self.manager.preprocessing_code
        datafile_structure = get_directory_structure(self.manager.input_data_folder)
        knowledge_key = iteration_type or "default"
        knowledge_pack = {}
        if hasattr(self.manager, "knowledge_packs"):
            knowledge_pack = self.manager.knowledge_packs.get(knowledge_key, {}) or {}
        input_contract_notes = knowledge_pack.get("input_contract_notes") or []
        
        if not preprocessing_code:
            logger.error("Preprocessing code not found. Cannot continue.")
            return {"status": "failed", "error": "Preprocessing code not available."}

        # 1. Generate modeling code
        decorator_chain = getattr(self.manager, "decorator_chain", None)
        prompt = self.prompt_handler.build(
            guideline=guideline,
            description=description,
            preprocessing_code=preprocessing_code,
            iteration_type=iteration_type,
            input_contract_notes=input_contract_notes,
            datafile_structure=datafile_structure,
            decorator_chain=decorator_chain,
        )
        
        # Save prompt for modeling
        self.manager.save_and_log_states(
            content=prompt,
            save_name="modeling/attempt_1/prompt.txt",
        )

        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(
            content=response,
            save_name="modeling/attempt_1/raw_response.txt",
        )

        modeling_code = self.prompt_handler.parse(response)
        # Save generated modeling code snapshot
        self.manager.save_and_log_states(
            content=modeling_code,
            save_name="modeling/attempt_1/generated_code.py",
        )

        self.manager.log_agent_end("Completed modeling code generation.")
        return {"status": "success", "code": modeling_code}