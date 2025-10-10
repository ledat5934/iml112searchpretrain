import json
import logging
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


def _infer_task_tag(task_type: Optional[str]) -> str:
    mapping = {
        "text_classification": "text-classification",
        "tabular_classification": "tabular-classification",
        "tabular_regression": "tabular-regression",
        "image_classification": "image-classification",
        "ner": "token-classification",
        "qa": "question-answering",
        "seq2seq": "text2text-generation",
    }
    return mapping.get((task_type or "").lower(), "text-classification")

class ModelRetrieverAgent(BaseAgent):
    """
    Retrieve candidate pretrained models for the problem using Google ADK SOTA search.
    Falls back to curated, offline suggestions when ADK is unavailable.
    """

    def __init__(self, config, manager, max_results: int = 5):
        super().__init__(config=config, manager=manager)
        self.max_results = max_results

    def _build_task_summary(self, desc: Dict[str, Any], prof: Optional[Dict[str, Any]] = None) -> str:
        task_type = desc.get("task_type") or "unknown"
        task = desc.get("task") or ""
        name = desc.get("name") or "dataset"
        files = ", ".join([f.get("name", "") for f in (prof.get("files") or [])][:5]) if prof else ""
        return (
            f"Dataset: {name}\n"
            f"Task: {task} ({task_type})\n"
            f"Files: {files}"
        )

    def _run_adk_search_blocking(self, task_summary: str, k: int, guideline: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Run ADK SOTA search ensuring explicit user_id/session_id and parse array JSON output."""
        try:
            from adk_search_sota import make_search_sota_root_agent
            from google.adk.runners import InMemoryRunner
            from google.genai import types as gen_types
            import asyncio, uuid, re as _re

            root_agent = make_search_sota_root_agent(task_summary=task_summary, k=k, guideline=guideline)
            runner = InMemoryRunner(agent=root_agent, app_name="sota-search")

            user_id = "manager"
            session_id = f"sota-{uuid.uuid4().hex[:8]}"
            user_msg = gen_types.Content(role="user", parts=[gen_types.Part(text="run")])

            def _strip_fences_txt(s: str) -> str:
                s = _re.sub(r"```+\w*\n", "", s)
                s = _re.sub(r"```+", "", s)
                return s

            async def _run_once():
                await runner.session_service.create_session(
                    app_name="sota-search",
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
                                looks_like_array = st.startswith("[") and ("model_name" in t2 or "example_code" in t2)
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
                    self.manager.save_and_log_states(items_json, "guideline/sota_search_raw.json", add_uuid=False)
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
                logger.error("SOTA search failed: no output produced.")
                return []
            if not parsed:
                logger.error("SOTA search failed: could not parse any valid model candidates.")
                return []

            try:
                preview = parsed[0].get("model_name", "") if isinstance(parsed, list) and parsed else ""
                logger.info(f"SOTA search returned {len(parsed)} candidates. First candidate: {preview}")
                self.manager.save_and_log_states(json.dumps(parsed, ensure_ascii=False, indent=2), "guideline/sota_search_parsed.json", add_uuid=False)
            except Exception:
                logger.info(f"SOTA search returned {len(parsed)} candidates.")

            return parsed
        except Exception as e:
            logger.error(f"SOTA search failed and is required to proceed: {e}")
            return []

    def __call__(self) -> Dict[str, Any]:
        self.manager.log_agent_start("ModelRetrieverAgent: retrieving pretrained SOTA models via ADK...")

        desc = getattr(self.manager, "description_analysis", {}) or {}
        prof = getattr(self.manager, "profiling_summary", {}) or {}
        task_summary = self._build_task_summary(desc, prof)

        # ADK SOTA search is a hard requirement; run with explicit session/user
        sota_models = self._run_adk_search_blocking(task_summary, k=self.max_results, guideline=None)

        # Filter models to allowed keys only
        allowed_keys = {"model_name", "example_code", "model_link"}
        cleaned_models: List[Dict[str, Any]] = []
        for m in (sota_models or []):
            try:
                cleaned = {k: m.get(k) for k in allowed_keys if m.get(k) is not None}
            except Exception:
                cleaned = {}
            if cleaned:
                cleaned_models.append(cleaned)

        # If no results, return error to stop iteration (no fallback)
        if not cleaned_models:
            self.manager.log_agent_end("ModelRetrieverAgent: no SOTA candidates — signaling failure.")
            return {"error": "sota_search_failed", "message": "No SOTA candidates parsed or available"}

        # Prepare suggestions payload with only requested fields
        suggestions: Dict[str, Any] = {
            "sota_models": cleaned_models,  # list of {model_name, example_code, model_link}
            "source": "sota-search",
            "note": "SOTA candidates via ADK SOTA search",
        }

        # Save to states on success only
        self.manager.save_and_log_states(
            json.dumps(suggestions, indent=2, ensure_ascii=False),
            "model_retrieval.json",
        )

        self.manager.log_agent_end("ModelRetrieverAgent: retrieval completed.")
        return suggestions
