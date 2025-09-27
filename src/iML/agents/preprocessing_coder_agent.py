import logging
import json
from typing import Dict, Any

from .base_agent import BaseAgent
from ..prompts import PreprocessingCoderPrompt
from .utils import init_llm

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
                iteration_type=iteration_type
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
                # Two-step LLM debug: summary (no search) -> refine (with google_search)
                filename = "code_generated"  # consistent with manager.execute_code script name
                task_desc = (self.manager.description_analysis or {}).get('task_description') or json.dumps(self.manager.description_analysis)
                ok, patched, meta = self.manager.debug_agent.llm_debug_fix(
                    code=code_to_execute,
                    stderr=error_message,
                    phase_name="preprocessing",
                    filename=filename,
                    attempt=attempt + 1,
                    task_description=task_desc,
                )
                if ok:
                    # DebugAgent already executed the refined code; no need to re-run
                    logger.info("Preprocessing code executed successfully after debug fixes (no re-run).")
                    self.manager.save_and_log_states(patched, "preprocessing/final_preprocessing_code.py")
                    self.manager.log_agent_end("Completed preprocessing code generation.")
                    return {"status": "success", "code": patched}
                # If not ok, continue with patched as next candidate to allow LLM loop to see updated code
                code_to_execute = patched

        logger.error(f"Unable to generate working preprocessing code after {self.max_retries} attempts.")
        self.manager.log_agent_end("Preprocessing code generation failed.")
        return {"status": "failed", "error": "Exceeded maximum retry attempts to generate code."}
