import json
import logging
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ArchitectureRetrieverAgent(BaseAgent):
    """
    Retrieve candidate neural network architectures for custom NN design using Google ADK search.
    """

    def __init__(self, config, manager, max_results: int = 5):
        super().__init__(config=config, manager=manager)
        self.max_results = max_results

    def _build_task_summary(self, desc: Dict[str, Any], prof: Optional[Dict[str, Any]] = None) -> str:
        """Build a task summary for architecture search."""
        task_type = desc.get("task_type") or "unknown"
        task = desc.get("task") or ""
        name = desc.get("name") or "dataset"
        
        # Extract key data characteristics
        data_shape = ""
        if prof:
            files = prof.get("files", [])
            if files:
                first_file = files[0]
                if isinstance(first_file, dict):
                    shape = first_file.get("shape", "")
                    if shape:
                        data_shape = f"\nData shape: {shape}"
        
        return (
            f"Dataset: {name}\n"
            f"Task: {task} ({task_type})"
            f"{data_shape}"
        )

    def _run_adk_search_blocking(self, task_summary: str, k: int) -> List[Dict[str, Any]]:
        """Run ADK Architecture search ensuring explicit user_id/session_id and parse array JSON output."""
        try:
            from adk_search_architecture import make_search_architecture_root_agent
            from google.adk.runners import InMemoryRunner
            from google.genai import types as gen_types
            import asyncio, uuid, re as _re

            root_agent = make_search_architecture_root_agent(task_summary=task_summary, k=k)
            runner = InMemoryRunner(agent=root_agent, app_name="architecture-search")

            user_id = "manager"
            session_id = f"arch-{uuid.uuid4().hex[:8]}"
            user_msg = gen_types.Content(role="user", parts=[gen_types.Part(text="run")])

            def _strip_fences_txt(s: str) -> str:
                s = _re.sub(r"```+\w*\n", "", s)
                s = _re.sub(r"```+", "", s)
                return s

            async def _run_once():
                await runner.session_service.create_session(
                    app_name="architecture-search",
                    user_id=user_id,
                    session_id=session_id,
                )

                items_json = None
                async for event in runner.run_async(
                    session_id=session_id,
                    user_id=user_id,
                    new_message=user_msg,
                ):
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if getattr(part, "text", None):
                                txt = part.text
                                t2 = _strip_fences_txt(txt)
                                st = t2.lstrip()
                                looks_like_array = st.startswith("[") and ("architecture_name" in t2 or "architecture_structure" in t2)
                                if looks_like_array:
                                    items_json = t2
                return items_json

            # Run coroutine respecting existing loop if present
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                items_json = asyncio.run(_run_once())
            else:
                items_json = loop.run_until_complete(_run_once())

            if items_json is not None:
                try:
                    # Save raw output for debugging
                    self.manager.save_and_log_states(items_json, "guideline/architecture_search_raw.json", add_uuid=False)
                except Exception:
                    pass

            parsed: List[Dict[str, Any]] = []
            if items_json:
                try:
                    parsed = json.loads(items_json)
                except Exception:
                    try:
                        cleaned = _strip_fences_txt(items_json)
                        parsed = json.loads(cleaned)
                    except Exception:
                        parsed = []

            if not items_json:
                logger.warning("Architecture search produced no output.")
                return []
            if not parsed:
                logger.warning("Architecture search could not parse any valid candidates.")
                return []

            try:
                preview = parsed[0].get("architecture_name", "") if isinstance(parsed, list) and parsed else ""
                logger.info(f"Architecture search returned {len(parsed)} candidate(s). First: {preview}")
                self.manager.save_and_log_states(json.dumps(parsed, ensure_ascii=False, indent=2), "guideline/architecture_search_parsed.json", add_uuid=False)
            except Exception:
                logger.info(f"Architecture search returned {len(parsed)} candidate(s).")

            return parsed
        except Exception as e:
            logger.warning(f"Architecture search failed: {e}")
            return []

    def __call__(self) -> Dict[str, Any]:
        self.manager.log_agent_start("ArchitectureRetrieverAgent: searching for suitable architectures via ADK...")

        desc = getattr(self.manager, "description_analysis", {}) or {}
        prof = getattr(self.manager, "profiling_summary", {}) or {}
        task_summary = self._build_task_summary(desc, prof)

        # ADK Architecture search (with fallback to None if fails)
        architecture_candidates = self._run_adk_search_blocking(task_summary, k=self.max_results)

        # Filter to allowed keys only
        allowed_keys = {"architecture_name", "architecture_structure", "source_link"}
        cleaned_architectures: List[Dict[str, Any]] = []
        for arch in (architecture_candidates or []):
            try:
                cleaned = {k: arch.get(k) for k in allowed_keys if arch.get(k) is not None}
            except Exception:
                cleaned = {}
            if cleaned:
                cleaned_architectures.append(cleaned)

        # If no results, return empty (will fallback to LLM design)
        if not cleaned_architectures:
            self.manager.log_agent_end("ArchitectureRetrieverAgent: no candidates found — will fallback to LLM design.")
            return {"architectures": [], "source": "architecture-search", "note": "Search failed, LLM will design from scratch"}

        # Prepare suggestions payload
        suggestions: Dict[str, Any] = {
            "architectures": cleaned_architectures,  # list of {architecture_name, architecture_structure, source_link}
            "source": "architecture-search",
            "note": "Architecture candidates via ADK search",
        }

        # Save to states
        self.manager.save_and_log_states(
            json.dumps(suggestions, indent=2, ensure_ascii=False),
            "architecture_retrieval.json",
        )

        self.manager.log_agent_end("ArchitectureRetrieverAgent: retrieval completed.")
        return suggestions

