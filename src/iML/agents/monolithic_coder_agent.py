import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from .base_agent import BaseAgent
from .utils import init_llm
from ..prompts import MonolithicCoderPrompt

logger = logging.getLogger(__name__)


class MonolithicCoderAgent(BaseAgent):
    """Agent that generates and executes a single monolithic script (preprocessing + modeling)."""

    def __init__(self, config: Dict, manager: Any, llm_config: Dict, max_retries: int = 10):
        super().__init__(config, manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="monolithic_coder",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = MonolithicCoderPrompt(manager=manager, llm_config=llm_config)
        self.max_retries = max_retries

    def __call__(self, iteration_type: str = None) -> Dict[str, Any]:
        self.manager.log_agent_start("Starting monolithic code generation...")

        guideline = getattr(self.manager, "guideline", {}) or {}
        description = self.manager.description_analysis or {}
        profiling_summary = getattr(self.manager, "profiling_summary", {}) or {}

        code_to_execute = None
        error_message = None
        submission_path = os.path.join(self.manager.output_folder, "submission.csv")

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"[MonolithicCoder] Attempt {attempt}/{self.max_retries}")
            decorator_chain = getattr(self.manager, "decorator_chain", None)
            prompt = self.prompt_handler.build(
                guideline=guideline,
                description=description,
                profiling_summary=profiling_summary,
                previous_code=code_to_execute,
                error_message=error_message,
                iteration_type=iteration_type,
                decorator_chain=decorator_chain,
            )
            self.manager.save_and_log_states(prompt, f"monolithic/attempt_{attempt}/prompt.txt")

            response = self.llm.assistant_chat(prompt)
            self.manager.save_and_log_states(response, f"monolithic/attempt_{attempt}/raw_response.txt")

            code_to_execute = self.prompt_handler.parse(response)
            self.manager.save_and_log_states(code_to_execute, f"monolithic/attempt_{attempt}/generated_code.py")

            execution_result = self.manager.execute_code(code_to_execute, "monolithic", attempt)
            if execution_result["success"]:
                if Path(submission_path).exists():
                    logger.info("Monolithic script executed successfully.")
                    self.manager.save_and_log_states(code_to_execute, "monolithic/final_code.py")
                    self.manager.log_agent_end("Completed monolithic code generation.")
                    return {"status": "success", "code": code_to_execute}

                error_message = "Execution reported success but submission.csv was missing."
                logger.error(error_message)
            else:
                error_message = execution_result["stderr"]
                logger.warning(f"[MonolithicCoder] Execution failed on attempt {attempt}.")

            if self.manager.is_debug_enabled():
                filename = "monolithic_script"
                task_desc = self.manager.build_debug_context(
                    stderr=error_message or "",
                    code=code_to_execute,
                    phase_name="monolithic",
                    attempt=attempt,
                )
                ok, patched, _ = self.manager.debug_agent.llm_debug_fix(
                    code=code_to_execute,
                    stderr=error_message or "",
                    phase_name="monolithic",
                    filename=filename,
                    attempt=attempt,
                    task_description=task_desc,
                    require_submission=True,
                    submission_filename="submission.csv",
                )
                if ok and Path(submission_path).exists():
                    logger.info("Monolithic script fixed via DebugAgent.")
                    self.manager.save_and_log_states(patched, "monolithic/final_code.py")
                    self.manager.log_agent_end("Completed monolithic code generation.")
                    return {"status": "success", "code": patched}
                if ok:
                    code_to_execute = patched

        logger.error("Unable to produce a working monolithic script.")
        self.manager.log_agent_end("Monolithic code generation failed.")
        return {"status": "failed", "error": error_message or "Exceeded retries."}

