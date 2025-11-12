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

    def __init__(self, config, manager, max_results: int = 1):
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

    def _validate_huggingface_model(self, model_link: str) -> bool:
        """Validate if a HuggingFace model exists by checking the model page."""
        if not model_link or "huggingface.co" not in model_link:
            logger.warning(f"Invalid HuggingFace link format: {model_link}")
            return False
        
        try:
            import requests
            response = requests.head(model_link, timeout=10, allow_redirects=True)
            
            # Only 200 OK is considered valid
            if response.status_code == 200:
                logger.debug(f"Model verified (200 OK): {model_link}")
                return True
            elif response.status_code == 404:
                logger.warning(f"Model not found (404): {model_link}")
                return False
            elif response.status_code == 401:
                logger.warning(f"Model unauthorized/private (401): {model_link}")
                return False  # Treat 401 as invalid - can't use private models
            elif response.status_code == 403:
                logger.warning(f"Model access forbidden (403): {model_link}")
                return False
            else:
                # For other status codes (500, 502, etc.), still reject
                logger.warning(f"Model validation failed (status {response.status_code}): {model_link}")
                return False
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout validating model (>10s): {model_link}")
            return False
        except Exception as e:
            logger.warning(f"Error validating model {model_link}: {e}")
            # Network errors - reject to be safe
            return False

    def __call__(self) -> Dict[str, Any]:
        self.manager.log_agent_start("ModelRetrieverAgent: retrieving pretrained SOTA models via ADK...")

        desc = getattr(self.manager, "description_analysis", {}) or {}
        prof = getattr(self.manager, "profiling_summary", {}) or {}
        task_summary = self._build_task_summary(desc, prof)

        # Retrieve more candidates (5x) to increase chances of finding valid models
        # ADK will return models in ranked order (best first)
        search_multiplier = 5
        total_to_search = self.max_results * search_multiplier
        logger.info(f"Searching for {total_to_search} model candidates to find top {self.max_results} valid model(s)...")
        
        sota_models = self._run_adk_search_blocking(task_summary, k=total_to_search, guideline=None)

        # Filter models to allowed keys only AND validate HuggingFace models
        # Keep only the first max_results valid models (they come ranked from ADK)
        allowed_keys = {"model_name", "example_code", "model_link"}
        cleaned_models: List[Dict[str, Any]] = []
        
        for idx, m in enumerate(sota_models or [], start=1):
            # Stop if we already have enough valid models
            if len(cleaned_models) >= self.max_results:
                logger.info(f"Found {self.max_results} valid model(s), stopping validation.")
                break
                
            try:
                cleaned = {k: m.get(k) for k in allowed_keys if m.get(k) is not None}
            except Exception:
                cleaned = {}
                
            if cleaned:
                model_name = cleaned.get('model_name', 'unknown')
                model_link = cleaned.get("model_link", "")
                
                logger.info(f"Validating candidate #{idx}: {model_name}...")
                
                # Validate HuggingFace model exists before adding
                if self._validate_huggingface_model(model_link):
                    cleaned_models.append(cleaned)
                    logger.info(f"✓ Valid model #{len(cleaned_models)}: {model_name}")
                else:
                    logger.info(f"✗ Skipping invalid/non-existent model: {model_name}")

        # If no results after validation, return empty suggestions (let LLM choose model itself)
        if not cleaned_models:
            logger.warning(f"Searched {total_to_search} candidates but found 0 valid models after validation.")
            logger.info("ModelRetrieverAgent: No valid SOTA candidates found — returning empty suggestions. LLM will choose model itself.")
            self.manager.log_agent_end("ModelRetrieverAgent: No valid models found, returning empty suggestions.")
            # Return empty suggestions instead of error to allow pipeline to continue
            suggestions: Dict[str, Any] = {
                "sota_models": [],  # Empty list
                "source": "sota-search",
                "note": "SOTA search found no valid candidates — LLM will choose model",
            }
            self.manager.save_and_log_states(
                json.dumps(suggestions, indent=2, ensure_ascii=False),
                "model_retrieval.json",
            )
            return suggestions

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
