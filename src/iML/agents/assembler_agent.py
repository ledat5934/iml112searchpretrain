# src/iML/agents/assembler_agent.py
import logging
import json
import os
from typing import Dict, Any

from .base_agent import BaseAgent
from ..prompts import AssemblerPrompt
from .utils import init_llm

logger = logging.getLogger(__name__)

class AssemblerAgent(BaseAgent):
    """
    Agent to assemble, finalize, execute and fix final code.
    """
    def __init__(self, config: Dict, manager: Any, llm_config: Dict, max_retries: int = 10):
        super().__init__(config, manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="assembler",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = AssemblerPrompt(
            manager=manager, 
            llm_config=self.llm_config
        )
        self.max_retries = max_retries

    def __call__(self, iteration_type=None) -> Dict[str, Any]:
        """
        Assemble, execute and retry final code until successful.
        """
        self.manager.log_agent_start("Starting assembly and testing of final code...")

        preprocessing_code = self.manager.preprocessing_code
        modeling_code = self.manager.modeling_code
        description = self.manager.description_analysis

        if not preprocessing_code or not modeling_code:
            error = "Preprocessing or modeling code not available."
            logger.error(error)
            return {"status": "failed", "error": error}

        # Combine initial code
        combined_code = preprocessing_code + "\n\n" + modeling_code
        submission_path = os.path.join(self.manager.output_folder, "submission.csv")
        error_message = None

        for attempt in range(self.max_retries):
            logger.info(f"Assembly and execution attempt {attempt + 1}/{self.max_retries}...")
            logger.detail(f"[ASSEMBLER] attempt={attempt+1} max_retries={self.max_retries} submission_path={submission_path}")
            if error_message:
                tail = str(error_message).splitlines()[-5:]
                logger.detail(f"[ASSEMBLER] attempt={attempt+1} previous_error_tail=\n" + "\n".join(tail))

            # 1. Assemble/Fix code
            # On first attempt, error_message is None, LLM will just assemble.
            # In subsequent attempts, LLM will fix errors.
            prompt = self.prompt_handler.build(
                original_code=combined_code,
                output_path=submission_path,
                description=description,
                error_message=error_message,
                iteration_type=iteration_type
            )

            # Save prompt under structured path
            self.manager.save_and_log_states(
                content=prompt,
                save_name=f"assemble/attempt_{attempt + 1}/prompt.txt",
            )

            response = self.llm.assistant_chat(prompt)
            self.manager.save_and_log_states(
                content=response,
                save_name=f"assemble/attempt_{attempt + 1}/raw_response.txt",
            )

            final_code = self.prompt_handler.parse(response)
            self.manager.save_and_log_states(
                content=final_code,
                save_name=f"assemble/attempt_{attempt + 1}/generated_code.py",
            )
            
            # 2. Execute final code
            execution_result = self.manager.execute_code(final_code, "assemble", attempt + 1)
            logger.detail(
                f"[ASSEMBLER] attempt={attempt+1} exec_success={bool(execution_result.get('success'))} "
                f"submission_exists={os.path.exists(submission_path)}"
            )
            
            if execution_result["success"]:
                # Treat as success only if submission.csv exists
                if os.path.exists(submission_path):
                    logger.info("Final code executed successfully!")
                    logger.info(f"Submission file created at: {submission_path}")
                    self.manager.save_and_log_states(final_code, "assemble/final_executable_code.py")
                    self.manager.log_agent_end("Completed assembly and execution of code.")
                    return {"status": "success", "code": final_code, "submission_path": submission_path}
                else:
                    logger.error("Execution returned success but submission.csv was not found. Will attempt debug fix.")
                    error_message = execution_result.get("stderr", "Missing submission.csv after run")
                    logger.info(f"[ASSEMBLER->DEBUG_AGENT] trigger=missing_submission attempt={attempt+1} debug_enabled={self.manager.is_debug_enabled()}")
                    # Invoke the same two-step LLM debug flow even when return code is 0 but artifact missing
                    last_10_lines = error_message.split('\n')[-10:]
                    error_to_log = '\n'.join(last_10_lines)
                    dataset_paths = (self.manager.description_analysis or {}).get('link to the dataset', [])
                    self.manager.save_and_log_states(
                        f"---ATTEMPT {attempt+1}---\nDATASET PATHS:\n{dataset_paths}\n\nCODE:\n{final_code}\n\nERROR:\n{error_to_log}",
                        f"assemble/attempt_{attempt+1}/failed.log"
                    )
                    if not self.manager.is_debug_enabled():
                        combined_code = final_code
                        continue

                    filename = "code_generated"
                    task_desc = self.manager.build_debug_context(
                        stderr=error_message,
                        code=final_code,
                        phase_name="assemble",
                        attempt=attempt + 1,
                    )
                    ok, patched, meta = self.manager.debug_agent.llm_debug_fix(
                        code=final_code,
                        stderr=error_message,
                        phase_name="assemble",
                        filename=filename,
                        attempt=attempt + 1,
                        task_description=task_desc,
                        require_submission=True,
                        submission_filename="submission.csv",
                    )
                    logger.info(f"[DEBUG_AGENT->ASSEMBLER] ok={ok} attempt={attempt+1} meta={meta}")
                    if ok:
                        # DebugAgent already ran the refined code successfully and enforced submission existence
                        if os.path.exists(submission_path):
                            logger.info("Final code executed successfully after debug fixes (no re-run).")
                            logger.info(f"Submission file created at: {submission_path}")
                            self.manager.save_and_log_states(patched, "assemble/final_executable_code.py")
                            self.manager.log_agent_end("Completed assembly and execution of code.")
                            return {"status": "success", "code": patched, "submission_path": submission_path}
                        else:
                            logger.error("DebugAgent reported success but submission.csv not found; continuing attempts.")
                    combined_code = patched if 'patched' in locals() else final_code
            # continue to next outer attempt
            else:
                error_message = execution_result["stderr"]
                last_10_lines = error_message.split('\n')[-10:]
                error_to_log = '\n'.join(last_10_lines)
                logger.warning(f"Code execution failed on attempt {attempt + 1}. Error: {error_message}")
                logger.info(f"[ASSEMBLER->DEBUG_AGENT] trigger=exec_failed attempt={attempt+1} debug_enabled={self.manager.is_debug_enabled()}")
                dataset_paths = (self.manager.description_analysis or {}).get('link to the dataset', [])
                self.manager.save_and_log_states(
                    f"---ATTEMPT {attempt+1}---\nDATASET PATHS:\n{dataset_paths}\n\nCODE:\n{final_code}\n\nERROR:\n{error_to_log}",
                    f"assemble/attempt_{attempt+1}/failed.log"
                )
                if not self.manager.is_debug_enabled():
                    combined_code = final_code
                    continue

                filename = "code_generated"
                task_desc = self.manager.build_debug_context(
                    stderr=error_message,
                    code=final_code,
                    phase_name="assemble",
                    attempt=attempt + 1,
                )
                ok, patched, meta = self.manager.debug_agent.llm_debug_fix(
                    code=final_code,
                    stderr=error_message,
                    phase_name="assemble",
                    filename=filename,
                    attempt=attempt + 1,
                    task_description=task_desc,
                    require_submission=True,
                    submission_filename="submission.csv",
                )
                logger.info(f"[DEBUG_AGENT->ASSEMBLER] ok={ok} attempt={attempt+1} meta={meta}")
                if ok:
                    # DebugAgent already ran refined code and ensured submission
                    if os.path.exists(submission_path):
                        logger.info("Final code executed successfully after debug fixes (no re-run).")
                        logger.info(f"Submission file created at: {submission_path}")
                        self.manager.save_and_log_states(patched, "assemble/final_executable_code.py")
                        self.manager.log_agent_end("Completed assembly and execution of code.")
                        return {"status": "success", "code": patched, "submission_path": submission_path}
                # Update combined_code for next LLM fix loop using the best patched
                combined_code = patched if 'patched' in locals() else final_code

        logger.error(f"Unable to generate working code after {self.max_retries} attempts.")
        self.manager.log_agent_end("Code assembly failed.")
        return {"status": "failed", "error": "Exceeded maximum retry attempts to generate code."}
