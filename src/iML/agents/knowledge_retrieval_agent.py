# src/iML/agents/knowledge_retrieval_agent.py
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from ..prompts.knowledge_retrieval_prompt import KnowledgeRetrievalPrompt
from .utils import init_llm

logger = logging.getLogger(__name__)

try:
    # Optional ADK tools for google_search-based knowledge retrieval
    from google.genai import types as adk_types
    from google.adk import agents as adk_agents
    from google.adk.tools.google_search_tool import google_search
    from google.adk.runners import InMemoryRunner

    ADK_AVAILABLE = True
except Exception:
    ADK_AVAILABLE = False


class KnowledgeRetrievalAgent(BaseAgent):
    """
    LLM-based agent that produces iteration-specific knowledge packs.
    No example code is allowed in the output.
    """

    def __init__(self, config, manager, llm_config, prompt_template=None):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.prompt_template = prompt_template
        self.prompt_handler = KnowledgeRetrievalPrompt(
            llm_config=self.llm_config,
            manager=self.manager,
            template=self.prompt_template,
        )
        self.llm = init_llm(
            llm_config=self.llm_config,
            agent_name="knowledge_retrieval_agent",
            multi_turn=self.llm_config.get("multi_turn", False),
        )

    def _build_task_summary(self) -> str:
        desc = getattr(self.manager, "description_analysis", {}) or {}
        prof = getattr(self.manager, "profiling_summary", {}) or {}
        task = (desc.get("task") or desc.get("task_description") or "").strip()
        name = (desc.get("name") or "dataset").strip()
        files = ""
        try:
            files_list = prof.get("files") or []
            if isinstance(files_list, list) and files_list:
                files = ", ".join([(f.get("name", "") if isinstance(f, dict) else str(f)) for f in files_list][:8])
        except Exception:
            files = ""
        parts = [
            f"Dataset: {name}",
            f"Task: {task}" if task else "",
        ]
        if files:
            parts.append(f"Files: {files}")
        return "\n".join(parts).strip()

    def _strip_fences(self, s: str) -> str:
        s = re.sub(r"```+\w*\n", "", s)
        s = re.sub(r"```+", "", s)
        return s

    def _extract_best_json_object(self, text: str) -> Optional[str]:
        """Pick a JSON object substring that looks like the required knowledge pack."""
        if not text:
            return None
        raw = self._strip_fences(text).strip()
        # Fast path: whole string is a JSON object
        if raw.startswith("{") and raw.endswith("}"):
            return raw
        # Heuristic: find the first {...} block that contains iteration_type
        candidates: List[str] = []
        for m in re.finditer(r"\{", raw):
            start = m.start()
            # bounded scan for a matching close brace (avoid huge scans)
            for end in range(min(len(raw) - 1, start + 50000), start, -1):
                if raw[end] == "}":
                    chunk = raw[start : end + 1]
                    if '"iteration_type"' in chunk or "'iteration_type'" in chunk:
                        candidates.append(chunk)
                        break
        if not candidates:
            return None
        # Return the longest candidate (usually the full object)
        return max(candidates, key=len)

    def _run_adk_with_search_blocking(self, prompt_text: str, save_suffix: str) -> Tuple[Optional[str], List[Dict[str, Any]], bool]:
        """Run an ADK agent that can call google_search; return (text, event_summaries, saw_search)."""
        if not ADK_AVAILABLE:
            return None, [], False
        try:
            import asyncio
            import uuid

            model_name = os.getenv("KNOWLEDGE_SEARCH_MODEL", "gemini-2.5-flash")

            def instruction_fn(ctx):
                return prompt_text

            agent = adk_agents.Agent(
                model=model_name,
                name="knowledge_retrieval_agent",
                description="Build iteration-specific ML knowledge pack; may use search to confirm best practices.",
                instruction=instruction_fn,
                tools=[google_search],
                generate_content_config=adk_types.GenerateContentConfig(temperature=0.2),
                include_contents="none",
            )

            root = adk_agents.SequentialAgent(
                name="knowledge_retrieval_root",
                description="Root wrapper for knowledge retrieval with search",
                sub_agents=[agent],
            )

            runner = InMemoryRunner(agent=root, app_name="knowledge-search")
            user_id = "manager"
            session_id = f"knowledge-{uuid.uuid4().hex[:8]}"
            user_msg = adk_types.Content(role="user", parts=[adk_types.Part(text="run")])

            async def _run_once():
                await runner.session_service.create_session(
                    app_name="knowledge-search",
                    user_id=user_id,
                    session_id=session_id,
                )
                out_text = ""
                event_summaries: List[Dict[str, Any]] = []
                saw_google_search = False
                async for event in runner.run_async(
                    session_id=session_id,
                    user_id=user_id,
                    new_message=user_msg,
                ):
                    try:
                        ev_dump = None
                        if hasattr(event, "model_dump"):
                            ev_dump = event.model_dump()
                        elif hasattr(event, "dict"):
                            ev_dump = event.dict()
                        if isinstance(ev_dump, dict):
                            # Best-effort: detect tool invocation by keyword
                            ev_json = json.dumps(ev_dump, ensure_ascii=False)
                            if "google_search" in ev_json:
                                saw_google_search = True
                            # Keep small summary (avoid giant dumps)
                            event_summaries.append(
                                {
                                    "event_type": ev_dump.get("type") or ev_dump.get("event_type") or type(event).__name__,
                                    "has_content": bool(ev_dump.get("content")),
                                }
                            )
                    except Exception:
                        pass

                    try:
                        if getattr(event, "content", None) and getattr(event.content, "parts", None):
                            for p in event.content.parts:
                                if getattr(p, "text", None):
                                    out_text += p.text
                    except Exception:
                        pass
                return out_text, event_summaries, saw_google_search

            # Run coroutine respecting existing loop if present
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                out_text, events, saw = asyncio.run(_run_once())
            else:
                out_text, events, saw = loop.run_until_complete(_run_once())

            # Save events + raw output for audit/debug
            try:
                self.manager.save_and_log_states(
                    json.dumps({"saw_google_search": saw, "events": events}, ensure_ascii=False, indent=2),
                    f"knowledge/knowledge_{save_suffix}_adk_events.json",
                )
            except Exception:
                pass
            try:
                self.manager.save_and_log_states(
                    out_text or "",
                    f"knowledge/knowledge_{save_suffix}_adk_raw_response.txt",
                )
            except Exception:
                pass

            return out_text, events, saw
        except Exception as e:
            logger.warning(f"KnowledgeRetrievalAgent: ADK run failed, fallback to LLM-only. Error: {e}")
            return None, [], False

    def __call__(self, iteration_type: str, model_suggestions: Dict[str, Any] | None = None, architecture_suggestions: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.manager.log_agent_start(f"KnowledgeRetrievalAgent: building knowledge pack ({iteration_type})...")

        task_context = getattr(self.manager, "task_context", {}) or {}
        if not task_context:
            logger.warning("KnowledgeRetrievalAgent: task_context missing; proceeding with empty context.")

        prompt = self.prompt_handler.build(
            iteration_type=iteration_type,
            task_context=task_context,
            model_suggestions=model_suggestions,
            architecture_suggestions=architecture_suggestions,
        )
        save_suffix = iteration_type or "default"
        # Add a compact task summary section to make retrieval more robust.
        task_summary = self._build_task_summary()
        if task_summary:
            prompt = (
                "# TASK SUMMARY\n"
                f"{task_summary}\n\n"
                + prompt
            )

        search_enabled = bool(getattr(self.manager, "is_search_enabled", lambda: True)())
        # Encourage tool usage only when search mode allows it.
        if search_enabled:
            prompt_for_search = (
                prompt
                + "\n\nIf you have access to a google_search tool, use it 2-4 times to confirm best practices and include 3-8 URLs/keywords under sources_or_keywords."
            )
        else:
            prompt_for_search = (
                prompt
                + "\n\nRun mode is LLM-only. Do NOT rely on external search/tools; infer best practices from provided context."
            )

        self.manager.save_and_log_states(prompt_for_search, f"knowledge/knowledge_{save_suffix}_prompt.txt")

        response = None
        # Prefer ADK+Search when available; fallback to backbone LLM.
        if ADK_AVAILABLE and search_enabled:
            logger.info(
                f"[KNOWLEDGE_RETRIEVAL] adk_available=True google_search_tool_enabled=True "
                f"model={os.getenv('KNOWLEDGE_SEARCH_MODEL', 'gemini-2.5-flash')} iteration_type={iteration_type}"
            )
            out_text, _events, saw_search = self._run_adk_with_search_blocking(prompt_for_search, save_suffix=save_suffix)
            response = out_text
            try:
                self.manager.save_and_log_states(
                    json.dumps({"adk_used": True, "saw_google_search": saw_search}, ensure_ascii=False, indent=2),
                    f"knowledge/knowledge_{save_suffix}_adk_summary.json",
                )
            except Exception:
                pass

        if not response:
            logger.info(
                f"[KNOWLEDGE_RETRIEVAL] adk_available={ADK_AVAILABLE} fallback_llm_provider={self.llm_config.get('provider')} "
                f"fallback_llm_model={self.llm_config.get('model')} iteration_type={iteration_type}"
            )
            response = self.llm.assistant_chat(prompt_for_search)
            self.manager.save_and_log_states(response, f"knowledge/knowledge_{save_suffix}_raw_response.txt")

        # Robust JSON extraction in case ADK returns extra text.
        extracted = self._extract_best_json_object(response or "")
        parsed_text = extracted if extracted else (response or "")
        knowledge_pack = self.prompt_handler.parse(parsed_text)
        try:
            self.manager.save_and_log_states(
                json.dumps(knowledge_pack, indent=2, ensure_ascii=False),
                f"knowledge/knowledge_{save_suffix}.json",
            )
        except Exception:
            pass

        self.manager.log_agent_end(f"KnowledgeRetrievalAgent: knowledge pack completed ({iteration_type}).")
        return knowledge_pack
