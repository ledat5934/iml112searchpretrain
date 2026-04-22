import logging

from rich.progress import (
    Progress,
    TextColumn,
)

from ..prompts import ExecuterPrompt
from ..utils.execution_env import build_child_execution_env
from ..utils.rich_logging import show_progress_bar
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


def execute_code(code, language, timeout):
    """
    Execute code with real-time output streaming and timeout and show a linear timeout progress bar..
    Args:
        code (str): The code to execute (Python code or bash script)
        language (str): The language to execute ("python" or "bash")
        timeout (float): Maximum execution time in seconds before terminating the process.
    Returns:
        tuple: (success: bool, stdout: str, stderr: str)
    """
    import select
    import subprocess
    import time
    import sys

    try:
        # Set up the command based on language
        if language.lower() == "python":
            cmd = [sys.executable, "-c", code]
        elif language.lower() == "bash":
            cmd = ["bash", "-c", code]
        else:
            return False, "", f"Unsupported language: {language}. Use 'python' or 'bash'."

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=build_child_execution_env(),
        )

        stdout_chunks, stderr_chunks = [], []

        # Set up tracking of both output streams
        streams = [process.stdout, process.stderr]

        # Track start time for timeout
        start_time = time.time()

        with Progress(
            TextColumn(f"[bold cyan]Executing {language}:"),
            TextColumn("[bold green]{task.completed:.1f}s[/bold green] [dim](time limit: {task.total:.0f}s)[/dim]"),
            refresh_per_second=2,
            transient=False,
            disable=not show_progress_bar(),
        ) as progress_context:

            task = progress_context.add_task("", total=timeout)

            while streams:
                # Calculate remaining time
                elapsed_time = time.time() - start_time
                progress_context.update(task, completed=elapsed_time)
                remaining_time = max(0, timeout - elapsed_time)

                # Check if we've exceeded timeout
                if remaining_time <= 0:
                    process.terminate()
                    time.sleep(3)  # Give it a moment to terminate gracefully
                    if process.poll() is None:  # If still running
                        process.kill()  # Force kill
                    stdout_chunks.append(f"\nProcess reached time limit after {timeout} seconds.\n")
                    logger.info(f"\nProcess reached time limit after {timeout} seconds.\n")
                    break

                # Wait for output on either stream with timeout
                # select.select returns empty lists if the timeout elapses
                readable, _, _ = select.select(streams, [], [], min(1, remaining_time))

                # If nothing was read but process is still running, continue the loop
                if not readable and process.poll() is None:
                    continue

                # If nothing was read and process exited, exit loop
                if not readable and process.poll() is not None:
                    break

                for stream in readable:
                    line = stream.readline()
                    if not line:  # EOF
                        streams.remove(stream)
                        continue

                    # Handle stdout
                    if stream == process.stdout:
                        stdout_chunks.append(line)
                        logger.detail(line.rstrip())
                    # Handle stderr
                    else:
                        stderr_chunks.append(line)
                        logger.detail(line.rstrip())

            elapsed_time = time.time() - start_time
            progress_context.update(task, completed=elapsed_time)

        # Wait for process to complete (should already be done, but just in case)
        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                stderr_chunks.append("Process forcibly terminated after timeout\n")

        success = process.returncode == 0
        return success, "".join(stdout_chunks), "".join(stderr_chunks)

    except Exception as e:
        return False, "", f"Error executing {language} code: {str(e)}"


class ExecuterAgent(BaseAgent):
    """
    Execute the code and give analysis.

    Agent Input:

    Agent Output:
    """

    def __init__(self, config, manager, language, timeout, executer_llm_config, executer_prompt_template):
        super().__init__(config=config, manager=manager)
        assert language in ["bash", "python"]

        self.timeout = timeout
        self.language = language
        self.executer_llm_config = executer_llm_config

        if executer_prompt_template is not None:
            self.executer_prompt_template = executer_prompt_template
        elif self.executer_llm_config.template is not None:
            self.executer_prompt_template = self.executer_llm_config.template
        else:
            self.executer_prompt_template = None

        if self.executer_llm_config.multi_turn:
            self.executer_llm = init_llm(
                llm_config=self.executer_llm_config,
                agent_name=f"{language}_executer",
                multi_turn=self.executer_llm_config.multi_turn,
            )

        self.executer_prompt = ExecuterPrompt(
            llm_config=self.executer_llm_config, manager=manager, template=self.executer_prompt_template
        )

        # Internal run counter to organize states/executer attempts
        self._run_counter = 0

    def _save_and_run(self, code: str, language: str, timeout: float):
        """Save code to states/executer/attempt_i/ then execute the saved file with streaming and timeout.

        Returns: (success: bool, stdout: str, stderr: str, attempt_dir: Path)
        """
        from pathlib import Path
        import os, sys, time, subprocess, select

        # Increment attempt counter
        self._run_counter += 1
        attempt_dir = Path(self.manager.output_folder) / "states" / "executer" / f"attempt_{self._run_counter}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        # Choose filenames by language
        if language.lower() == "python":
            script_path = attempt_dir / "code.py"
        else:
            script_path = attempt_dir / "code.sh"

        stdout_path = attempt_dir / "stdout.txt"
        stderr_path = attempt_dir / "stderr.txt"

        # Save code first
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        logger.info(f"ExecuterAgent: executing saved file at {script_path}")

        # Build command
        if language.lower() == "python":
            cmd = [sys.executable, str(script_path)]
        elif language.lower() == "bash":
            cmd = ["bash", str(script_path)]
        else:
            return False, "", f"Unsupported language: {language}. Use 'python' or 'bash'.", attempt_dir

        # Working directory consistent with Manager: dataset root (input_data_folder)
        try:
            from pathlib import Path as _P
            working_dir = str(_P(self.manager.input_data_folder).parent)
        except Exception:
            working_dir = None

        # Run with streaming and timeout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=working_dir,
            env=build_child_execution_env(),
        )

        stdout_chunks, stderr_chunks = [], []
        streams = [process.stdout, process.stderr]
        start_time = time.time()

        while streams:
            elapsed = time.time() - start_time
            remaining = max(0, timeout - elapsed)
            if remaining <= 0:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                stderr_chunks.append(f"\nProcess reached time limit after {timeout} seconds.\n")
                logger.error(f"Process reached time limit after {timeout} seconds.")
                break

            readable, _, _ = select.select([s for s in streams if s], [], [], min(1, remaining))
            if not readable and process.poll() is not None:
                break
            for stream in readable:
                line = stream.readline()
                if not line:
                    streams.remove(stream)
                    continue
                if stream is process.stdout:
                    stdout_chunks.append(line)
                    logger.detail(line.rstrip())
                else:
                    stderr_chunks.append(line)
                    logger.detail(line.rstrip())

        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                stderr_chunks.append("Process forcibly terminated after timeout\n")

        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)

        # Persist outputs next to the code
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(stdout)
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(stderr)

        return process.returncode == 0, stdout, stderr, attempt_dir

    def __call__(self, code_to_execute, code_to_analyze=None, task_description=None, data_prompt=None):

        self.manager.log_agent_start("ExecuterAgent: executing code and collecting stdout/stderr for evaluation.")

        if code_to_analyze is None:
            code_to_analyze = code_to_execute

        # Save then run the saved code (sequential, not inline -c)
        success, stdout, stderr, attempt_dir = self._save_and_run(
            code=code_to_execute, language=self.language, timeout=self.timeout
        )

        if not self.executer_llm_config.multi_turn:
            self.executer_llm = init_llm(
                llm_config=self.executer_llm_config,
                agent_name=f"{self.language}_executer",
                multi_turn=self.executer_llm_config.multi_turn,
            )

        # Build prompt for evaluating execution results
        prompt = self.executer_prompt.build(
            stdout=stdout,
            stderr=stderr,
            python_code=code_to_analyze,
            task_description=task_description,
            data_prompt=data_prompt,
        )

        # Query the LLM
        response = self.executer_llm.assistant_chat(prompt)

        # Parse the LLM response to extract decision and error summary
        decision, error_summary = self.executer_prompt.parse(response)

        # Log the decision and error summary
        logger.brief(f"Planner decision: {decision}")
        if error_summary:
            logger.info(f"Error summary: {error_summary}")

        self.manager.log_agent_end("ExecuterAgent: execution finished; planner decision logged.")

        return decision, error_summary, prompt, stderr, stdout
