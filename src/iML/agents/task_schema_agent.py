# src/iML/agents/task_schema_agent.py
import json
import logging
from pathlib import Path
from typing import Dict, Any

from .base_agent import BaseAgent
from ..prompts.task_schema_prompt import TaskSchemaPrompt
from ..utils.file_io import get_directory_structure
from .utils import init_llm

logger = logging.getLogger(__name__)


class TaskSchemaAgent(BaseAgent):
    """
    LLM-based agent that infers a task schema from description + profiling evidence.
    """

    def __init__(self, config, manager, llm_config, prompt_template=None):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.prompt_template = prompt_template
        self.prompt_handler = TaskSchemaPrompt(
            llm_config=self.llm_config,
            manager=self.manager,
            template=self.prompt_template,
        )
        self.llm = init_llm(
            llm_config=self.llm_config,
            agent_name="task_schema_agent",
            multi_turn=self.llm_config.get("multi_turn", False),
        )

    def __call__(self) -> Dict[str, Any]:
        self.manager.log_agent_start("TaskSchemaAgent: Inferring task schema...")

        dataset_path = self.manager.input_data_folder
        description_file_path = Path(dataset_path) / "description.txt"
        if not description_file_path.exists():
            logger.error(f"TaskSchemaAgent: description.txt not found at {description_file_path}")
            return {"error": "description.txt not found"}

        try:
            with open(description_file_path, "r", encoding="utf-8") as f:
                description_text = f.read()
        except Exception as e:
            logger.error(f"TaskSchemaAgent: failed to read description.txt: {e}")
            return {"error": f"failed to read description.txt: {e}"}

        description_analysis = getattr(self.manager, "description_analysis", {}) or {}
        profiling_summary = getattr(self.manager, "profiling_summary", {}) or {}
        directory_structure = get_directory_structure(dataset_path)

        prompt = self.prompt_handler.build(
            description_text=description_text,
            description_analysis=description_analysis,
            profiling_summary=profiling_summary,
            directory_structure=directory_structure,
        )

        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, "task_schema_raw_response.txt")

        task_schema = self.prompt_handler.parse(response)
        try:
            self.manager.save_and_log_states(
                json.dumps(task_schema, indent=2, ensure_ascii=False),
                "task_schema.json",
            )
        except Exception:
            pass

        self.manager.log_agent_end("TaskSchemaAgent: Task schema inference completed.")
        return task_schema
