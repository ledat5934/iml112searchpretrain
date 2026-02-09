import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)

PHASES = ("guideline", "preprocessing", "modeling", "assembler")


class BasePromptDecorator(ABC):

    @abstractmethod
    def get_guideline_section(self) -> str:
        return ""

    @abstractmethod
    def get_preprocessing_section(self) -> str:
        return ""

    @abstractmethod
    def get_modeling_section(self) -> str:
        return ""

    @abstractmethod
    def get_assembler_section(self) -> str:
        return ""

    @abstractmethod
    def get_algorithm_constraint(self) -> str:
        return ""

    @abstractmethod
    def get_batch_config(self) -> tuple:
        return ("", "")

    @abstractmethod
    def get_data_handling_section(self) -> str:
        return ""

    def get_section(self, phase: str) -> str:
        mapping = {
            "guideline": self.get_guideline_section,
            "preprocessing": self.get_preprocessing_section,
            "modeling": self.get_modeling_section,
            "assembler": self.get_assembler_section,
        }
        fn = mapping.get(phase)
        if fn:
            return fn()
        return ""


class DomainDecorator(BasePromptDecorator):

    def get_algorithm_constraint(self) -> str:
        return ""

    def get_batch_config(self) -> tuple:
        return ("", "")

    def get_data_handling_section(self) -> str:
        return ""

    def get_assembler_section(self) -> str:
        return ""


class TaskDecorator(BasePromptDecorator):

    def get_algorithm_constraint(self) -> str:
        return ""

    def get_batch_config(self) -> tuple:
        return ("", "")

    def get_data_handling_section(self) -> str:
        return ""

    def get_assembler_section(self) -> str:
        return ""


class DecoratorChain:

    def __init__(
        self,
        domain_decorators: List[DomainDecorator] = None,
        task_decorators: List[TaskDecorator] = None,
    ):
        self.domain_decorators: List[DomainDecorator] = domain_decorators or []
        self.task_decorators: List[TaskDecorator] = task_decorators or []

    def _collect(self, phase: str) -> str:
        parts = []
        for d in self.domain_decorators:
            s = d.get_section(phase)
            if s and s.strip():
                parts.append(s)
        for t in self.task_decorators:
            s = t.get_section(phase)
            if s and s.strip():
                parts.append(s)
        return "\n\n".join(parts)

    def get_combined_section(self, phase: str) -> str:
        return self._collect(phase)

    def get_domain_sections(self, phase: str) -> str:
        parts = []
        for d in self.domain_decorators:
            s = d.get_section(phase)
            if s and s.strip():
                parts.append(s)
        return "\n\n".join(parts)

    def get_task_sections(self, phase: str) -> str:
        parts = []
        for t in self.task_decorators:
            s = t.get_section(phase)
            if s and s.strip():
                parts.append(s)
        return "\n\n".join(parts)

    def get_algorithm_constraints(self) -> str:
        parts = []
        for d in self.domain_decorators:
            c = d.get_algorithm_constraint()
            if c and c.strip():
                parts.append(c)
        for t in self.task_decorators:
            c = t.get_algorithm_constraint()
            if c and c.strip():
                parts.append(c)
        return "\n".join(parts)

    def get_merged_batch_config(self) -> tuple:
        instructions = []
        formats = []
        for d in self.domain_decorators:
            bi, df = d.get_batch_config()
            if bi and bi.strip():
                instructions.append(bi)
            if df and df.strip():
                formats.append(df)
        for t in self.task_decorators:
            bi, df = t.get_batch_config()
            if bi and bi.strip():
                instructions.append(bi)
            if df and df.strip():
                formats.append(df)
        return ("\n".join(instructions), "\n".join(formats))

    def get_merged_data_handling(self) -> str:
        parts = []
        for d in self.domain_decorators:
            s = d.get_data_handling_section()
            if s and s.strip():
                parts.append(s)
        for t in self.task_decorators:
            s = t.get_data_handling_section()
            if s and s.strip():
                parts.append(s)
        return "\n\n".join(parts)

    def wrap_prompt(self, base_prompt: str, phase: str) -> str:
        domain_text = self.get_domain_sections(phase)
        task_text = self.get_task_sections(phase)
        extra = ""
        if domain_text:
            extra += f"\n\n## DOMAIN-SPECIFIC GUIDANCE\n{domain_text}"
        if task_text:
            extra += f"\n\n## TASK-SPECIFIC GUIDANCE\n{task_text}"
        return base_prompt + extra

    def is_empty(self) -> bool:
        return len(self.domain_decorators) == 0 and len(self.task_decorators) == 0

    def summary(self) -> str:
        domains = [type(d).__name__ for d in self.domain_decorators]
        tasks = [type(t).__name__ for t in self.task_decorators]
        return f"DecoratorChain(domains={domains}, tasks={tasks})"
