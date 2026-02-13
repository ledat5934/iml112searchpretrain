import logging
import json
from typing import Dict, Any

from .base_agent import BaseAgent
from ..prompts import PreprocessingCoderPrompt
from .utils import init_llm
from ..utils.file_io import get_directory_structure

logger = logging.getLogger(__name__)

class PreprocessingCoderAgent(BaseAgent):
    """
    Agent to create and execute preprocessing code.
    It has a retry loop to generate and validate code until it runs successfully or runs out of retries.
    """
    def __init__(self, config: Dict, manager: Any, llm_config: Dict, max_retries: int = 10):
        super().__init__(config, manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="preprocessing_coder",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = PreprocessingCoderPrompt(
            manager=manager, 
            llm_config=self.llm_config
        )
        self.max_retries = max_retries

    def __call__(self, iteration_type=None) -> Dict[str, Any]:
        """
        Generate, execute and retry preprocessing code until successful or maximum retries exceeded.
        """
        self.manager.log_agent_start("Starting preprocessing code generation...")

        guideline = self.manager.guideline
        description = self.manager.description_analysis
        datafile_structure = get_directory_structure(self.manager.input_data_folder)
        knowledge_key = iteration_type or "default"
        knowledge_pack = {}
        if hasattr(self.manager, "knowledge_packs"):
            knowledge_pack = self.manager.knowledge_packs.get(knowledge_key, {}) or {}
        input_contract_notes = knowledge_pack.get("input_contract_notes") or []
        model_input_data = knowledge_pack.get("model_input_data") or []
        decided_fields = {}
        try:
            decided_all = self.manager.get_prompt_fields(iteration_type)
            decided_fields = (decided_all or {}).get("preprocessing", {}) or {}
        except Exception:
            decided_fields = {}
        
        code_to_execute = None
        error_message = None
        
        for attempt in range(self.max_retries):
            logger.info(f"Code generation attempt {attempt + 1}/{self.max_retries}...")

            # 1. Generate code
            prompt = self.prompt_handler.build(
                guideline=guideline,
                description=description,
                previous_code=code_to_execute,
                error_message=error_message,
                iteration_type=iteration_type,
                input_contract_notes=input_contract_notes,
                model_input_data=model_input_data,
                datafile_structure=datafile_structure,
                prompt_fields=decided_fields,
            )

            # Save prompt under structured path
            self.manager.save_and_log_states(
                content=prompt,
                save_name=f"preprocessing/attempt_{attempt + 1}/prompt.txt",
            )

            response = self.llm.assistant_chat(prompt)
            self.manager.save_and_log_states(
                content=response,
                save_name=f"preprocessing/attempt_{attempt + 1}/raw_response.txt",
            )
            
            code_to_execute = self.prompt_handler.parse(response)
            # Save generated code snapshot for this attempt
            self.manager.save_and_log_states(
                content=code_to_execute,
                save_name=f"preprocessing/attempt_{attempt + 1}/generated_code.py",
            )

            # 2. Execute code
            execution_result = self.manager.execute_code(code_to_execute, "preprocessing", attempt + 1)
            
            # 3. Check results
            if execution_result["success"]:
                logger.info("Preprocessing code executed successfully!")
                self.manager.save_and_log_states(code_to_execute, "preprocessing/final_preprocessing_code.py")
                self.manager.log_agent_end("Completed preprocessing code generation.")
                return {"status": "success", "code": code_to_execute}
            else:
                error_message = execution_result["stderr"]
                last_10_lines = error_message.split('\n')[-10:]
                error_to_log = '\n'.join(last_10_lines)
                logger.warning(f"Code execution failed on attempt {attempt + 1}. Error: {error_to_log}")
                dataset_paths = (self.manager.description_analysis or {}).get('link to the dataset', [])
                self.manager.save_and_log_states(
                    f"---ATTEMPT {attempt+1}---\nDATASET PATHS:\n{dataset_paths}\n\nCODE:\n{code_to_execute}\n\nERROR:\n{error_to_log}",
                    f"preprocessing/attempt_{attempt+1}/failed.log"
                )
                if not self.manager.is_debug_enabled():
                    logger.info("DebugAgent disabled in static ablation mode; retrying via LLM.")
                    continue

                filename = "code_generated"  # consistent with manager.execute_code script name
                task_desc = self.manager.build_debug_context(
                    stderr=error_message,
                    code=code_to_execute,
                    phase_name="preprocessing",
                    attempt=attempt + 1,
                )
                ok, patched, meta = self.manager.debug_agent.llm_debug_fix(
                    code=code_to_execute,
                    stderr=error_message,
                    phase_name="preprocessing",
                    filename=filename,
                    attempt=attempt + 1,
                    task_description=task_desc,
                    datafile_structure=datafile_structure,
                )
                if ok:
                    logger.info("Preprocessing code executed successfully after debug fixes (no re-run).")
                    self.manager.save_and_log_states(patched, "preprocessing/final_preprocessing_code.py")
                    self.manager.log_agent_end("Completed preprocessing code generation.")
                    return {"status": "success", "code": patched}
                code_to_execute = patched

        logger.error(f"Unable to generate working preprocessing code after {self.max_retries} attempts.")
        self.manager.log_agent_end("Preprocessing code generation failed.")
        return {"status": "failed", "error": "Exceeded maximum retry attempts to generate code."}
