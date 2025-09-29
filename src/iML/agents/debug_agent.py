import re
import os
import json
import difflib
from typing import Dict, Any, List, Tuple, Optional

from .base_agent import BaseAgent
from ..llm import ChatLLMFactory

try:
    # Optional ADK tools for google_search-based refine
    from google.genai import types as adk_types
    from google.adk import agents as adk_agents
    from google.adk.tools.google_search_tool import google_search
    from google.adk.runners import InMemoryRunner
    ADK_AVAILABLE = True
except Exception:
    ADK_AVAILABLE = False


class DebugAgent(BaseAgent):
    """
    Heuristic debug fixer: parses stderr, builds quick queries, synthesizes a fix plan,
    applies code patches, runs, and loops up to N rounds.
    """

    def __init__(self, config, manager, max_rounds: int = 5):
        super().__init__(config=config, manager=manager)
        self.max_rounds = max_rounds
        # Prompts for LLM-based debug
        self.BUG_SUMMARY_INSTR = (
            """# Error report
{bug}

# Description analysis (JSON)
{description_json}

# Your task
- Remove all unnecessary parts of the above error report.
- We are now running {filename}.py. Do not remove where the error occurred.

# Output constraint
{submission_path_note}
"""
        )
        self.BUG_REFINE_INSTR = (
            """# Task description
{task_description}

# Description analysis (JSON)
{description_json}

# Code with an error:
{code}

# Error:
{bug}

# Output constraint
{submission_path_note}

# Your task
- Please revise the code to fix the error.
- If the error is a 'module not found` error, then install the necessary module. You can use `pip install <module>`, where `<module>` is the name of the module to install.
- Do not remove subsampling if exists.
- Provide the improved, self-contained Python script again.
- There should be no additional headings or text in your response.
- Remember to print a line in the code with 'Final Validation Performance: {final_validation_score}' so we can parse performance.
- The code should be a single-file python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the refined Python code."""
        )

    # ── LLM step A: Summarize error (no search) ─────────────────────
    def _get_description(self) -> dict:
        """Return description_analysis from manager or load from states if missing."""
        desc = getattr(self.manager, 'description_analysis', None)
        if desc:
            return desc
        # Fallback: try load from states file
        try:
            states_dir = os.path.join(self.manager.output_folder, "states")
            path = os.path.join(states_dir, "description_analyzer_response.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _llm_bug_summary(self, stderr: str, filename: str, phase_name: str, attempt_index: int) -> str:
        # Choose an LLM config available (reuse assembler)
        llm_config = getattr(self.manager.assembler_agent, 'llm_config', None)
        if llm_config is None:
            raise RuntimeError("LLM config not available for bug summary.")
        # Pull description from manager to include context and build prompt
        description = self._get_description() or {}
        try:
            description_json = json.dumps(description, ensure_ascii=False, indent=2)
        except Exception:
            description_json = json.dumps(description, indent=2)
        submission_path_note = ""
        if phase_name == "assemble":
            expected_name = "submission.csv"
            try:
                expected_abs = os.path.join(getattr(self.manager, 'output_folder', '.'), expected_name)
            except Exception:
                expected_abs = expected_name
            submission_path_note = f"If a submission file is produced, it MUST be saved to this absolute path: {expected_abs}."

        prompt = self.BUG_SUMMARY_INSTR.format(
            bug=stderr,
            filename=filename,
            description_json=description_json,
            submission_path_note=submission_path_note,
        )
        # Run single-turn summary
        chat = ChatLLMFactory.get_chat_model(llm_config, session_name=f"single_turn_bug_summary_{phase_name}")
        chat.initialize_conversation(chat, system_prompt="You are a concise debugging assistant.")
        summary = chat.assistant_chat(prompt)
        # Save into the step/attempt folder
        save_name = f"{phase_name}/attempt_{attempt_index}/bug_summary.txt"
        self.manager.save_and_log_states(summary, save_name)
        return summary

    # ── LLM step B: Refine code (with google_search tool) ───────────
    def _extract_code_block(self, text: str) -> Optional[str]:
        # Prefer ```python ... ``` or any ``` ... ```
        m = re.search(r"```python\s*\n(.*?)```", text, flags=re.S|re.I)
        if m:
            return m.group(1).strip()
        m = re.search(r"```\s*\n(.*?)```", text, flags=re.S)
        if m:
            return m.group(1).strip()
        return None

    def _llm_refine_code(self, task_description: str, code: str, bug_summary: str, phase_name: str, attempt_index: int) -> tuple[str, str, str]:
        # Pull description from manager to include context
        description = self._get_description() or {}
        try:
            description_json = json.dumps(description, ensure_ascii=False, indent=2)
        except Exception:
            description_json = json.dumps(description, indent=2)
        raw_text = ""
        # If assembling, enforce explicit absolute submission path in the prompt
        submission_path_note = ""
        if phase_name == "assemble":
            expected_name = "submission.csv"
            try:
                expected_abs = os.path.join(getattr(self.manager, 'output_folder', '.'), expected_name)
            except Exception:
                expected_abs = expected_name
            submission_path_note = f"If you produce a submission file, you MUST save it to this absolute path: {expected_abs}."

        prompt_text = self.BUG_REFINE_INSTR.format(
            task_description=task_description or "",
            description_json=description_json,
            code=code,
            bug=bug_summary,
            final_validation_score="{final_validation_score}",
            submission_path_note=submission_path_note,
        )
        if ADK_AVAILABLE:
            # Use ADK runner with a single Agent having google_search tool
            import asyncio, uuid

            def instruction_fn(ctx):
                return prompt_text

            model_name = os.getenv("BUG_FIX_MODEL", "gemini-2.5-flash")
            agent = adk_agents.Agent(
                model=model_name,
                name="bug_refine_agent",
                description="Refine Python code to fix errors, may use search to consult docs.",
                instruction=instruction_fn,
                tools=[google_search],
                generate_content_config=adk_types.GenerateContentConfig(temperature=0.2),
                include_contents="none",
            )

            # Wrap in a Sequential root agent
            root = adk_agents.SequentialAgent(
                name="bug_refine_root",
                description="Root wrapper for bug refine with search",
                sub_agents=[agent],
            )

            runner = InMemoryRunner(agent=root, app_name="bug-refine")
            user_id = "debugger"
            session_id = f"bugfix-{uuid.uuid4().hex[:8]}"

            async def _run_once():
                await runner.session_service.create_session(
                    app_name="bug-refine", user_id=user_id, session_id=session_id
                )
                out_text = ""
                user_msg = adk_types.Content(role="user", parts=[adk_types.Part(text="run")])
                async for event in runner.run_async(
                    session_id=session_id, user_id=user_id, new_message=user_msg
                ):
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if getattr(part, "text", None):
                                out_text += part.text or ""
                return out_text

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                out_text = asyncio.run(_run_once())
            else:
                out_text = loop.run_until_complete(_run_once())

            raw_text = out_text
            code_block = self._extract_code_block(raw_text)
            refined_code = code_block or raw_text
        else:
            # Fallback: plain LLM without tools
            llm_config = getattr(self.manager.assembler_agent, 'llm_config', None)
            if llm_config is None:
                raise RuntimeError("LLM config not available for bug refine.")
            chat = ChatLLMFactory.get_chat_model(llm_config, session_name="single_turn_bug_refine")
            chat.initialize_conversation(chat, system_prompt="You are a senior Python engineer.")
            out = chat.assistant_chat(prompt_text)
            raw_text = out
            code_block = self._extract_code_block(raw_text)
            refined_code = code_block or raw_text

        # Save prompt and raw response into the step/attempt folder
        self.manager.save_and_log_states(prompt_text, f"{phase_name}/attempt_{attempt_index}/prompt.txt")
        self.manager.save_and_log_states(raw_text, f"{phase_name}/attempt_{attempt_index}/raw_response.txt")
        return refined_code, raw_text, prompt_text

    def llm_debug_fix(self, code: str, stderr: str, phase_name: str, filename: str, attempt: int, task_description: Optional[str], require_submission: bool = False, submission_filename: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """Two-stage LLM debug: summarize (no search) then refine (with google_search); loop up to max_rounds.

        If require_submission is True (used for assemble), success requires both returncode==0 and the
        existence of the expected submission file (defaults to 'submission.csv' in manager.output_folder).
        """
        current = code
        for round_idx in range(1, self.max_rounds + 1):
            # The bug summary must be stored with the code that produced the error
            bug_attempt_index = attempt + (round_idx - 1)  # first round -> current failed attempt
            # 1) Summarize bug into the failed attempt folder
            bug_summary = self._llm_bug_summary(stderr, filename, phase_name, bug_attempt_index)
            # 2) Refine code for the next attempt
            refined_attempt_index = bug_attempt_index + 1
            refined, raw_text, prompt_text = self._llm_refine_code(task_description or "", current, bug_summary, phase_name, refined_attempt_index)
            if not refined or len(refined.strip()) < 5:
                # nothing returned, abort
                return False, current, {"rounds": round_idx, "reason": "empty_refine", "last_attempt_index": refined_attempt_index}
            # Lightweight health checks before running
            hc_notes: List[str] = []
            if "if __name__ == \"__main__\":" not in refined:
                hc_notes.append("missing __main__ block")
            if phase_name == "assemble" and require_submission:
                # Heuristic: check code references the expected output path
                expected_name = (submission_filename or "submission.csv")
                if expected_name not in refined:
                    hc_notes.append(f"code does not reference {expected_name}")

            # Save refined code snapshot into the attempt folder
            self.manager.save_and_log_states(refined, f"{phase_name}/attempt_{refined_attempt_index}/generated_code.py")
            # Run refined code
            result = self.manager.execute_code(refined, f"{phase_name}", refined_attempt_index)
            ok = bool(result.get("success"))
            if ok and require_submission and phase_name == "assemble":
                # Require artifact existence
                out_dir = getattr(self.manager, 'output_folder', None) or "."
                expected = submission_filename or "submission.csv"
                ok = os.path.exists(os.path.join(out_dir, expected))
            # Debug round logging (short)
            tail = (result.get("stderr") or "").splitlines()[-10:]
            tail_text = "\n".join(tail)
            status_text = "ok" if ok else "fail"
            self.manager.save_and_log_states(
                f"round {round_idx} -> {status_text}. stderr available at states/{phase_name}/attempt_{refined_attempt_index}/stderr.txt\nHealth checks: {', '.join(hc_notes) if hc_notes else 'passed'}",
                f"{phase_name}/attempt_{refined_attempt_index}/round_summary.txt"
            )
            if ok:
                return True, refined, {"rounds": round_idx, "last_result": result, "last_attempt_index": refined_attempt_index}
            # Prepare next round
            current = refined
            stderr = result.get("stderr", stderr)
        # If we exhausted rounds, last attempt index is attempt + self.max_rounds
        return False, current, {"rounds": self.max_rounds, "last_stderr": stderr, "last_attempt_index": attempt + self.max_rounds}

    # ── Step 1: Parse error ─────────────────────────────────────────
    def parse_error(self, stderr: str) -> Dict[str, Any]:
        exc_type = None
        message = None
        file_line = None
        traceback_tail = None

        if not stderr:
            return {"exc_type": None, "message": None, "file_line": None, "traceback_tail": None}

        lines = stderr.strip().splitlines()
        # Grab last non-empty line as primary error line
        tail_line = ""
        for line in reversed(lines):
            if line.strip():
                tail_line = line.strip()
                break
        traceback_tail = "\n".join(lines[-10:])  # last 10 lines context

        # Patterns: "ModuleNotFoundError: No module named 'x'"
        m = re.search(r"^(\w+Error):\s*(.*)$", tail_line)
        if m:
            exc_type = m.group(1)
            message = m.group(2)

        # File line like "File \"path\", line 123, in <module>"
        for line in reversed(lines):
            fm = re.search(r"File\s+\"(.+?)\",\s+line\s+(\d+)", line)
            if fm:
                file_line = f"{fm.group(1)}:{fm.group(2)}"
                break

        return {
            "exc_type": exc_type,
            "message": message,
            "file_line": file_line,
            "traceback_tail": traceback_tail,
        }

    # ── Step 2: Make queries ────────────────────────────────────────
    def make_queries(self, err: Dict[str, Any]) -> List[str]:
        q: List[str] = []
        et, msg = (err.get("exc_type") or "").strip(), (err.get("message") or "").strip()
        if et:
            q.append(f"{et} {msg} python".strip())
        # Extract potential lib or attribute tokens from message
        lib = None
        attr = None
        pkg_match = re.search(r"named\s+'([^']+)'|No module named '?([\w-]+)'?", msg)
        if pkg_match:
            lib = pkg_match.group(1) or pkg_match.group(2)
        attr_match = re.search(r"attribute\s+'([^']+)'|got an unexpected keyword argument '([^']+)'", msg)
        if attr_match:
            attr = attr_match.group(1) or attr_match.group(2)
        if lib and et:
            q.append(f"{lib} {et} {attr or ''} python".strip())
        # Some canned patterns
        if et == "ModuleNotFoundError" and lib:
            q.append(f"ModuleNotFoundError: {lib} python how to install")
        if et == "TypeError" and attr:
            q.append(f"scikit-learn got an unexpected keyword argument '{attr}'")
        if et == "AttributeError" and attr:
            q.append(f"pandas '{attr}' replacement")
        return [s for s in q if s]

    # ── Step 3-4: Synthesize fix plan using heuristics ───────────────
    def synthesize_fix_plan(self, err: Dict[str, Any]) -> List[Dict[str, Any]]:
        et = (err.get("exc_type") or "").strip()
        msg = (err.get("message") or "").strip()
        plan: List[Dict[str, Any]] = []

        # Module not found -> install and import
        m = re.search(r"No module named '([^']+)'|No module named ([\w-]+)", msg)
        if et == "ModuleNotFoundError" and m:
            pkg = m.group(1) or m.group(2)
            plan.append({"action": "ensure_package", "package": pkg})
            return plan

        # Unexpected kwarg -> remove kwarg in calls
        m = re.search(r"unexpected keyword argument '([^']+)'", msg)
        if et == "TypeError" and m:
            kw = m.group(1)
            plan.append({"action": "remove_kwarg", "kwarg": kw})
            return plan

        # AttributeError common replacements
        m = re.search(r"attribute '([^']+)'", msg)
        if et == "AttributeError" and m:
            attr = m.group(1)
            mapping = {
                "as_matrix": "to_numpy",
                "ix": "loc",
                "append": None,  # encourage refactor but we attempt minimal change
            }
            repl = mapping.get(attr)
            if repl is not None:
                plan.append({"action": "replace_attr", "old": attr, "new": repl})
            else:
                plan.append({"action": "hint_replace_attr", "old": attr})
            return plan

        # FileNotFoundError: suggest ensure directories
        if et == "FileNotFoundError":
            plan.append({"action": "path_fix_hint"})
            return plan

        # ValueError shape -> ravel y
        if et == "ValueError" and ("inconsistent" in msg or "shape" in msg or "Expected 2D array" in msg):
            plan.append({"action": "reshape_hint"})
            return plan

        # Syntax/Indentation: try simple fix hint
        if et in ("SyntaxError", "IndentationError"):
            plan.append({"action": "format_hint"})
            return plan

        # Fallback: no-op (let LLM or next layer handle)
        plan.append({"action": "noop"})
        return plan

    # ── Step 5: Apply patch ──────────────────────────────────────────
    def apply_fix(self, code: str, plan: List[Dict[str, Any]], dataset_paths: List[str]) -> str:
        patched = code
        for step in plan:
            act = step.get("action")
            if act == "ensure_package":
                pkg = step.get("package")
                patched = self._ensure_package_import(patched, pkg)
            elif act == "remove_kwarg":
                kw = step.get("kwarg")
                patched = self._remove_kwarg(patched, kw)
            elif act == "replace_attr":
                old, new = step.get("old"), step.get("new")
                patched = re.sub(rf"\.{old}(\s*\()", f".{new}(", patched)
            elif act == "hint_replace_attr":
                # No safe auto-fix; leave as-is
                pass
            elif act == "path_fix_hint":
                # Provide dataset paths as comment near top if missing
                if "# DATASET_PATHS" not in patched:
                    header = f"\n# DATASET_PATHS: {json.dumps(dataset_paths)}\n"
                    patched = header + patched
            elif act == "reshape_hint":
                # Add a helper hint comment; real reshape requires context
                if "# DEBUG_HINT: reshape" not in patched:
                    patched = "# DEBUG_HINT: reshape y with y.ravel() if needed\n" + patched
            elif act == "format_hint":
                # No automatic reformat here
                pass
            elif act == "noop":
                pass

        return patched

    def _ensure_package_import(self, code: str, package: str) -> str:
        # Insert try/except import with pip install at top if missing
        import_block = (
            f"\nimport sys, subprocess\n"
            f"try:\n"
            f"    __import__('{package}')\n"
            f"except ModuleNotFoundError:\n"
            f"    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '{package}'])\n"
            f"    __import__('{package}')\n"
        )
        # Place after shebang or at very top
        if code.startswith("#!"):
            idx = code.find("\n") + 1
            return code[:idx] + import_block + code[idx:]
        else:
            return import_block + code

    def _remove_kwarg(self, code: str, kw: str) -> str:
        # Remove occurrences of 'kw=...,' inside function calls (heuristic)
        pattern = re.compile(rf"(\b\w+\s*\(.*?){kw}\s*=\s*[^,)]+,?", re.DOTALL)

        def repl(m):
            s = m.group(0)
            # remove kw=... possibly with trailing comma; keep call structure
            s = re.sub(rf"{kw}\s*=\s*[^,)]+,?\s*", "", s)
            # fix double commas or stray commas before )
            s = re.sub(r",\s*\)", ")", s)
            s = re.sub(r"\(\s*,", "(", s)
            return s

        return pattern.sub(repl, code)

    # ── Step 6-7: Run and loop ───────────────────────────────────────
    def debug_fix(self, code: str, stderr: str, phase_name: str, dataset_paths: List[str]) -> Tuple[bool, str, Dict[str, Any]]:
        current = code
        for i in range(1, self.max_rounds + 1):
            err = self.parse_error(stderr)
            queries = self.make_queries(err)
            plan = self.synthesize_fix_plan(err)

            # Save debug artifacts
            self.manager.save_and_log_states(json.dumps({"round": i, "err": err, "queries": queries, "plan": plan}, ensure_ascii=False, indent=2), f"debug_{phase_name}_round_{i}.json")

            patched = self.apply_fix(current, plan, dataset_paths)
            if patched != current:
                diff = difflib.unified_diff(current.splitlines(), patched.splitlines(), fromfile="before.py", tofile="after.py", lineterm="")
                self.manager.save_and_log_states("\n".join(diff), f"debug_{phase_name}_round_{i}.diff")

            # Run patched code
            result = self.manager.execute_code(patched, f"{phase_name}_debug", i)
            if result.get("success"):
                return True, patched, {"rounds": i, "last_result": result}

            # Prepare next round
            current = patched
            stderr = result.get("stderr", stderr)

        return False, current, {"rounds": self.max_rounds, "last_stderr": stderr}
