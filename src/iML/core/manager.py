import logging
import sys
import os
import uuid
import subprocess
import json
import shutil
import time
import signal
import threading
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from omegaconf import OmegaConf

from ..agents import (
    DescriptionAnalyzerAgent,
    ProfilingAgent,
    ProfilingLLMAgent,
    ProfilingSummarizerAgent,
    ModelRetrieverAgent,
    ArchitectureRetrieverAgent,
    TaskSchemaAgent,
    KnowledgeRetrievalAgent,
    GuidelineAgent,
    PreprocessingCoderAgent,
    ModelingCoderAgent,
    MonolithicCoderAgent,
    AssemblerAgent,
    ComparisonAgent,
    DebugAgent,
    PromptDeciderAgent,
    DeploymentAgent,
)
from ..agents.comparison_agent import IterationResultExtractor
from ..llm import ChatLLMFactory
from ..utils.execution_env import build_child_execution_env
from ..utils.file_io import get_directory_structure
from ..utils.hw_monitor import HardwareMonitor

# Basic configuration
logging.basicConfig(level=logging.INFO)

# Create a logger
logger = logging.getLogger(__name__)

class IterationTimeoutError(Exception):
    """Custom exception for iteration timeout."""
    pass

def iteration_timeout_handler(signum, frame):
    """Signal handler for iteration timeout (Unix/Linux only)."""
    raise IterationTimeoutError("Iteration execution exceeded the time limit.")

class IterationTimer:
    """Cross-platform iteration timer using threading."""
    
    def __init__(self, timeout_seconds, callback):
        self.timeout_seconds = timeout_seconds
        self.callback = callback
        self.timer = None
        self.is_expired = False
    
    def start(self):
        """Start the timeout timer."""
        self.is_expired = False
        self.timer = threading.Timer(self.timeout_seconds, self._timeout_occurred)
        self.timer.start()
    
    def cancel(self):
        """Cancel the timeout timer."""
        if self.timer:
            self.timer.cancel()
    
    def _timeout_occurred(self):
        """Internal method called when timeout occurs."""
        self.is_expired = True
        self.callback()
    
    def check_timeout(self):
        """Check if timeout has occurred."""
        if self.is_expired:
            raise IterationTimeoutError("Iteration execution exceeded the time limit.")


def _run_single_iteration_worker(worker_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process worker: run one iteration with shared analysis context already prepared."""
    iteration = worker_payload["iteration"]
    original_output_folder = worker_payload["original_output_folder"]
    iteration_output = worker_payload["iteration_output"]
    iteration_timeout = int(worker_payload["iteration_timeout"])
    monitor_cfg = worker_payload.get("monitor_cfg", {}) or {}
    shared = worker_payload.get("shared_context", {}) or {}

    config = OmegaConf.create(worker_payload["config_dict"])
    manager = Manager(
        input_data_folder=worker_payload["input_data_folder"],
        output_folder=iteration_output,
        config=config,
        ablation_variant=worker_payload.get("ablation_variant"),
        search_mode=worker_payload.get("search_mode"),
        parallel_iterations=False,
    )

    # Inject shared analysis outputs (computed once in parent process).
    manager.description_analysis = shared.get("description_analysis") or {}
    manager.profiling_result = shared.get("profiling_result") or {}
    manager.profiling_summary = shared.get("profiling_summary") or {}
    manager.task_schema = shared.get("task_schema")
    manager.task_context = shared.get("task_context")

    Path(iteration_output).mkdir(parents=True, exist_ok=True)
    manager.output_folder = iteration_output

    iteration_start_time = time.time()
    hw_monitor = None
    iteration_status = "unknown"
    success = False

    if bool(monitor_cfg.get("enabled", True)):
        try:
            hw_monitor = HardwareMonitor(
                sample_interval_sec=float(monitor_cfg.get("sample_interval_sec", 1.0)),
                include_gpu=bool(monitor_cfg.get("include_gpu", True)),
                monitor_scope=str(monitor_cfg.get("monitor_scope", "process")),
                target_root_pid=int(monitor_cfg.get("target_root_pid", os.getpid())),
            )
            hw_monitor.start(iteration_name=iteration["name"])
        except Exception:
            hw_monitor = None

    try:
        success = manager._run_iteration_with_timeout(iteration["name"], iteration_timeout)
        iteration_status = "success" if success else "failed"
    except IterationTimeoutError:
        success = False
        iteration_status = "timeout"
    except Exception as e:
        success = False
        iteration_status = f"error: {e}"
    finally:
        duration_sec = round(time.time() - iteration_start_time, 3)
        if hw_monitor is not None:
            try:
                hw_file = os.path.join(iteration_output, "hardware_usage.json")
                hw_monitor.stop_and_export(
                    output_json_path=hw_file,
                    run_id=Path(original_output_folder).name,
                    iteration_name=iteration["name"],
                    extra_metadata={
                        "iteration_folder": iteration["folder"],
                        "iteration_description": iteration["description"],
                        "result_status": iteration_status,
                        "timeout_sec": iteration_timeout,
                        "duration_sec": duration_sec,
                        "execution_mode": "parallel_process",
                    },
                )
            except Exception:
                pass
        try:
            manager.cleanup()
        except Exception:
            pass

    return {
        "iteration_name": iteration["name"],
        "iteration_folder": iteration["folder"],
        "iteration_output": iteration_output,
        "status": iteration_status,
        "success": success,
    }


class Manager:
    def __init__(
        self,
        input_data_folder: str,
        output_folder: str,
        config: str,
        ablation_variant: str = None,
        search_mode: str | None = None,
        parallel_iterations: bool | None = None,
    ):
        """Initialize Manager with required paths and config from YAML file.

        Args:
            input_data_folder: Path to input data directory
            output_folder: Path to output directory
            config_path: Path to YAML configuration file
        """
        self.input_data_folder = input_data_folder
        self.output_folder = output_folder
        self.config = config
        self.ablation_variant = ablation_variant
        # Provide an instance logger for prompts/agents that expect manager.logger
        self.logger = logging.getLogger(__name__)
        # Search mode:
        # - "hybrid" (default): allow external retrieval/search tools (ADK) when available
        # - "llm_only": disable external search entirely; rely on backbone LLM only
        # Precedence: CLI/search_mode arg > config.search_mode; then ablation aliases can force llm_only if CLI unset.
        configured_search_mode = str(getattr(self.config, "search_mode", "hybrid") or "hybrid").strip().lower()
        if configured_search_mode not in {"hybrid", "llm_only"}:
            configured_search_mode = "hybrid"
        if search_mode is not None:
            cli_sm = str(search_mode).strip().lower()
            if cli_sm in {"hybrid", "llm_only"}:
                configured_search_mode = cli_sm
        elif (self.ablation_variant or "").strip().lower() in {"llm_only", "no_search", "nosearch"}:
            configured_search_mode = "llm_only"
        self.search_mode = configured_search_mode
        self.logger.info(f"[RUN_MODE] search_mode={self.search_mode}")

        configured_parallel = bool(getattr(self.config, "parallel_iterations", False))
        if parallel_iterations is None:
            self.parallel_iterations = configured_parallel
        else:
            self.parallel_iterations = bool(parallel_iterations)
        self.logger.info(f"[RUN_MODE] parallel_iterations={self.parallel_iterations}")

        # Validate paths
        for path, name in [(input_data_folder, "input_data_folder")]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.description_analyzer_agent = DescriptionAnalyzerAgent(
            config=config,
            manager=self,
            llm_config=self.config.description_analyzer,
        )
        self.profiling_agent = ProfilingAgent(
            config=config,
            manager=self,
        )
        profiling_llm_cfg = getattr(self.config, "profiling_llm_agent", None) or self.config.guideline_generator
        self.profiling_llm_agent = ProfilingLLMAgent(
            config=config,
            manager=self,
            llm_config=profiling_llm_cfg,
        )
        self.profiling_summarizer_agent = ProfilingSummarizerAgent(
            config=config,
            manager=self,
            llm_config=self.config.profiling_summarizer,
        )
        self.model_retriever_agent = ModelRetrieverAgent(
            config=config,
            manager=self,
        )
        self.architecture_retriever_agent = ArchitectureRetrieverAgent(
            config=config,
            manager=self,
        )
        task_schema_llm_config = getattr(self.config, "task_schema_agent", None) or self.config.guideline_generator
        self.task_schema_agent = TaskSchemaAgent(
            config=config,
            manager=self,
            llm_config=task_schema_llm_config,
        )
        knowledge_llm_config = getattr(self.config, "knowledge_retriever", None) or self.config.guideline_generator
        self.knowledge_retrieval_agent = KnowledgeRetrievalAgent(
            config=config,
            manager=self,
            llm_config=knowledge_llm_config,
        )
        self.guideline_agent = GuidelineAgent(
            config=config,
            manager=self,
            llm_config=self.config.guideline_generator,
        )
        prompt_decider_llm = getattr(self.config, "prompt_decider_agent", None) or self.config.guideline_generator
        self.prompt_decider_agent = PromptDeciderAgent(
            config=config,
            manager=self,
            llm_config=prompt_decider_llm,
        )
        self.preprocessing_coder_agent = PreprocessingCoderAgent(
            config=config,
            manager=self,
            llm_config=self.config.preprocessing_coder,
        )
        self.modeling_coder_agent = ModelingCoderAgent(
            config=config,
            manager=self,
            llm_config=self.config.modeling_coder,
        )
        monolithic_llm_config = getattr(self.config, "monolithic_coder", None)
        if monolithic_llm_config is None:
            monolithic_llm_config = self.config.modeling_coder
        self.monolithic_coder_agent = MonolithicCoderAgent(
            config=config,
            manager=self,
            llm_config=monolithic_llm_config,
        )
        self.assembler_agent = AssemblerAgent(
            config=config,
            manager=self,
            llm_config=self.config.assembler,
        )
        self.comparison_agent = ComparisonAgent(
            config=config,
            manager=self,
            llm_config=self.config.assembler,  # Using same LLM config as assembler
        )
        deployment_llm = getattr(self.config, "deployment_agent", None) or self.config.assembler
        self.deployment_agent = DeploymentAgent(
            config=config,
            manager=self,
            llm_config=deployment_llm,
        )
        # Initialize DebugAgent once for search-driven patching across phases
        self.debug_agent = DebugAgent(
            config=config,
            manager=self,
            max_rounds=1,
        )

        self.context = {
            "input_data_folder": input_data_folder,
            "output_folder": output_folder,
            
        }
        self.task_schema = None
        self.task_context = None
        self.knowledge_packs = {}
        self.prompt_fields_by_iteration: Dict[str, Any] = {}

    def _get_hardware_monitoring_config(self) -> Dict[str, Any]:
        """Read hardware monitoring settings from config with safe defaults."""
        hm_cfg = getattr(self.config, "hardware_monitoring", None)
        enabled = True
        sample_interval_sec = 1.0
        include_gpu = True

        if hm_cfg is not None:
            enabled = bool(getattr(hm_cfg, "enabled", enabled))
            sample_interval_sec = float(getattr(hm_cfg, "sample_interval_sec", sample_interval_sec))
            include_gpu = bool(getattr(hm_cfg, "include_gpu", include_gpu))

        return {
            "enabled": enabled,
            "sample_interval_sec": sample_interval_sec,
            "include_gpu": include_gpu,
        }

    def _run_deployment_stage(self, iteration_type: str | None = None) -> bool:
        """Run post-assembly deployment validation/refactor stage."""
        try:
            result = self.deployment_agent(iteration_type=iteration_type)
            if result.get("status") == "success":
                logger.info("DeploymentAgent completed successfully.")
                return True
            logger.error(f"DeploymentAgent failed: {result.get('error', 'unknown error')}")
            return False
        except Exception as e:
            logger.error(f"DeploymentAgent crashed: {e}")
            return False

    def get_prompt_fields(self, iteration_type: str | None = None) -> Dict[str, Any]:
        """
        Get (and lazily build) prompt field overrides for a given iteration.
        Output is a JSON-like dict with keys: preprocessing/modeling/assembler.
        """
        key = iteration_type or "default"
        existing = (self.prompt_fields_by_iteration or {}).get(key)
        if existing:
            return existing

        # Build using PromptDeciderAgent (best-effort; falls back to empty dict)
        knowledge_pack = {}
        try:
            if hasattr(self, "knowledge_packs"):
                knowledge_pack = self.knowledge_packs.get(key, {}) or {}
        except Exception:
            knowledge_pack = {}

        try:
            decided = self.prompt_decider_agent(
                iteration_type=iteration_type or "default",
                guideline=getattr(self, "guideline", {}) or {},
                knowledge_pack=knowledge_pack,
            )
            if isinstance(decided, dict) and "error" not in decided:
                self.prompt_fields_by_iteration[key] = decided
                return decided
        except Exception as e:
            logger.warning(f"PromptDeciderAgent failed: {e}")

        self.prompt_fields_by_iteration[key] = {}
        return {}

    # ------------------------------------------------------------------
    # Ablation helpers
    # ------------------------------------------------------------------
    def is_ablation_variant(self, variant_name: str = None) -> bool:
        if not self.ablation_variant:
            return False
        if variant_name is None:
            return True
        return self.ablation_variant == variant_name

    def is_monolithic_mode(self) -> bool:
        return self.is_ablation_variant("mono")

    def is_static_mode(self) -> bool:
        return self.is_ablation_variant("static")

    def is_debug_enabled(self) -> bool:
        return not self.is_static_mode()

    def is_search_enabled(self) -> bool:
        """Whether external web/ADK search is allowed in this run."""
        return (self.search_mode or "hybrid") != "llm_only"

    def _is_planning_result_valid(self, result: Any) -> bool:
        """Heuristic validator for planning-stage JSON outputs."""
        if not isinstance(result, dict):
            return False
        if not result:
            return False
        if result.get("status") == "failed":
            return False
        if "error" in result:
            return False
        return True

    def _run_planning_agent_with_retries(
        self,
        agent_name: str,
        call_fn,
        max_retries: int = 2,
        valid_fn=None,
    ) -> Dict[str, Any]:
        """Run a planning-style agent with retry on malformed/error outputs.

        max_retries=2 means up to 3 total attempts.
        """
        validator = valid_fn or self._is_planning_result_valid
        total_attempts = max(1, int(max_retries) + 1)
        last_result: Any = None
        last_error: str = "unknown"

        for attempt in range(1, total_attempts + 1):
            try:
                result = call_fn()
                last_result = result
                if validator(result):
                    if attempt > 1:
                        logger.info(f"[{agent_name}] succeeded on retry attempt {attempt}/{total_attempts}.")
                    return result
                last_error = "invalid_or_unparseable_json_output"
                logger.warning(
                    f"[{agent_name}] invalid result at attempt {attempt}/{total_attempts}; "
                    f"retrying..." if attempt < total_attempts else f"[{agent_name}] exhausted retries."
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"[{agent_name}] exception at attempt {attempt}/{total_attempts}: {e}"
                )

        if isinstance(last_result, dict):
            return {**last_result, "error": last_result.get("error") or last_error}
        return {"error": last_error}

    def _prepare_guideline(self, iteration_type: str = None) -> bool:
        """
        Generate or synthesize a guideline depending on the ablation variant.
        Returns True if guideline context is ready.
        """
        if self.is_ablation_variant("reactive"):
            logger.info("Ablation (reactive): Skipping GuidelineAgent and synthesizing minimal guidance.")
            self.guideline = self._build_reactive_guideline(iteration_type)
            try:
                serialized = json.dumps(self.guideline, ensure_ascii=False, indent=2)
                self.save_and_log_states(serialized, "guideline/guideline_response.json")
            except Exception:
                pass
            return True

        guideline = self._run_planning_agent_with_retries(
            "GuidelineAgent",
            lambda: self.guideline_agent(iteration_type=iteration_type),
            max_retries=2,
        )
        if "error" in guideline:
            logger.error(f"Guideline generation failed: {guideline['error']}")
            return False
        self.guideline = guideline
        logger.info(f"Guideline generated successfully for {iteration_type or 'default'}.")
        try:
            serialized = json.dumps(guideline, ensure_ascii=False, indent=2)
            self.save_and_log_states(serialized, "guideline/guideline_response.json")
        except Exception:
            pass
        return True

    def _build_reactive_guideline(self, iteration_type: str = None) -> Dict[str, Any]:
        """Construct a lightweight guideline structure from description/profiling outputs."""
        description = self.description_analysis or {}
        profiling_summary = getattr(self, "profiling_summary", {}) or {}
        profiling_result = getattr(self, "profiling_result", {}) or {}
        id_analysis = profiling_result.get("id_format_analysis", {})
        submission_analysis = id_analysis.get("submission_format_analysis") if id_analysis else None
        has_extensions = bool((submission_analysis or {}).get("submission_has_extensions"))
        id_column_name = (submission_analysis or {}).get("first_column_name", "id")

        modeling_section = {
            "note": "GuidelineAgent disabled. Infer modeling strategy directly from task description.",
            "iteration_type": iteration_type or "unspecified",
            "IDs_in_submission_file_contain_file_extensions": has_extensions,
            "create_submission_file": {
                "id_column_name": id_column_name,
                "notes": "Follow the dataset's sample submission exactly. No additional blueprint available."
            },
            "raw_description": description,
        }

        preprocessing_section = {
            "note": "No curated preprocessing plan. Use dataset description and profiling summary to design steps.",
            "profiling_summary": profiling_summary,
            "dataset_paths": description.get("link to the dataset", []),
        }

        target_info = profiling_result.get("target_info") or {
            "note": "Reactive mode could not infer structured target info. Deduce target column from dataset metadata."
        }

        return {
            "meta": {
                "ablation_variant": "iMLreactive",
                "iteration_type": iteration_type,
            },
            "preprocessing": preprocessing_section,
            "modeling": modeling_section,
            "target_identification": target_info,
        }

    def get_iteration_timeout(self, iteration_type):
        """Get the execution timeout for a specific iteration type."""
        # Check if iteration_timeouts configuration exists
        if hasattr(self.config, 'iteration_timeouts') and self.config.iteration_timeouts:
            timeout = self.config.iteration_timeouts.get(iteration_type)
            if timeout:
                return timeout
        
        # Fallback to default_iteration_timeout if configured
        if hasattr(self.config, 'default_iteration_timeout'):
            return self.config.default_iteration_timeout
            
        # Final fallback to per_execution_timeout
        return self.config.per_execution_timeout
    
    def _run_iteration_with_timeout(self, iteration_type, iteration_timeout):
        """Run iteration with cross-platform timeout handling."""
        is_windows = platform.system() == "Windows"
        
        if is_windows:
            # Use threading-based timeout for Windows
            timeout_occurred = threading.Event()
            
            def timeout_callback():
                timeout_occurred.set()
            
            timer = IterationTimer(iteration_timeout, timeout_callback)
            timer.start()
            
            try:
                # Run the iteration pipeline with periodic timeout checks
                success = self._run_iteration_pipeline_with_checks(iteration_type, timeout_occurred)
                return success
            finally:
                timer.cancel()
                
        else:
            # Use signal-based timeout for Unix/Linux
            original_handler = signal.signal(signal.SIGALRM, iteration_timeout_handler)
            signal.alarm(iteration_timeout)
            
            try:
                success = self._run_iteration_pipeline(iteration_type)
                return success
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)
    
    def _run_iteration_pipeline_with_checks(self, iteration_type, timeout_occurred):
        """Run iteration pipeline with periodic timeout checks for Windows."""
        try:
            # Step 1a: For custom_nn_search iteration, run architecture search
            if iteration_type == "custom_nn_search":
                if timeout_occurred.is_set():
                    raise IterationTimeoutError("Iteration timeout occurred before architecture search")
                architecture_suggestions = self._run_planning_agent_with_retries(
                    "ArchitectureRetrieverAgent",
                    lambda: self.architecture_retriever_agent(),
                    max_retries=2,
                )
                if architecture_suggestions and "error" not in architecture_suggestions and architecture_suggestions.get("architectures"):
                    self.architecture_suggestions = architecture_suggestions
                    logger.info("Architecture search completed for custom_nn_search iteration.")
                else:
                    # Fallback: continue without search (LLM designs from scratch)
                    logger.warning("Architecture search failed, LLM will design from scratch.")
                    self.architecture_suggestions = None
            else:
                # Ensure other iterations (including custom_nn) are not influenced by architecture search
                if hasattr(self, "architecture_suggestions"):
                    delattr(self, "architecture_suggestions")
            
            # Step 1b: For pretrained iteration, run model retrieval now; otherwise clear suggestions
            if iteration_type == "pretrained":
                if timeout_occurred.is_set():
                    raise IterationTimeoutError("Iteration timeout occurred before model retrieval")
                model_suggestions = self._run_planning_agent_with_retries(
                    "ModelRetrieverAgent",
                    lambda: self.model_retriever_agent(),
                    max_retries=2,
                )
                
                # Check for critical errors (not just empty results)
                if not model_suggestions:
                    logger.error("Pretrained iteration aborted: model_retriever_agent returned None.")
                    return False
                if "error" in model_suggestions:
                    logger.error(f"Pretrained iteration aborted: {model_suggestions.get('message', 'Unknown error')}")
                    return False
                    
                self.model_suggestions = model_suggestions
                models = self.model_suggestions.get("sota_models", [])
                
                # If no valid candidates found, fall back to LLM choosing model itself
                if not models:
                    logger.warning("No valid SOTA candidates found. Falling back to single guideline generation (LLM will choose model).")
                    # Continue with standard flow (non-candidate loop) - skip to line 339
                else:
                    # Build iteration-specific knowledge pack before candidate loop
                    self._prepare_iteration_knowledge(iteration_type)
                    # Candidate-wise loop: for each SOTA model, run guideline -> preprocessing -> modeling -> assembly
                    any_success = False
                    parent_iter_dir = Path(self.output_folder)
                    first_success_idx = None
                    for idx, cand in enumerate(models, start=1):
                        if timeout_occurred.is_set():
                            raise IterationTimeoutError("Iteration timeout occurred before guideline generation")
                        candidate_success = False

                        # Narrow suggestions to a single candidate
                        self.model_suggestions = {"sota_models": [cand], "source": model_suggestions.get("source", "sota-search")}

                        # Prepare candidate-specific output directory and switch context
                        candidate_dir = parent_iter_dir / f"candidate_{idx}"
                        candidate_dir.mkdir(parents=True, exist_ok=True)
                        original_output_folder = self.output_folder
                        self.output_folder = str(candidate_dir)

                        # Guideline
                        if not self._prepare_guideline(iteration_type=iteration_type):
                            logger.error(f"Guideline generation failed for candidate {idx}.")
                            self.output_folder = str(original_output_folder)
                            continue

                        if self.is_monolithic_mode():
                            if timeout_occurred.is_set():
                                raise IterationTimeoutError("Iteration timeout occurred before monolithic generation")
                            mono_result = self.monolithic_coder_agent(iteration_type=iteration_type)
                            if mono_result.get("status") == "failed":
                                logger.error(f"Monolithic generation failed for candidate {idx}: {mono_result.get('error')}")
                                self.output_folder = str(original_output_folder)
                                continue
                            candidate_success = True
                            self.assembled_code = mono_result.get("code")
                            if not self._run_deployment_stage(iteration_type=iteration_type):
                                logger.error(f"Deployment stage failed for candidate {idx}.")
                                candidate_success = False
                        else:
                            # Preprocessing
                            if timeout_occurred.is_set():
                                raise IterationTimeoutError("Iteration timeout occurred before preprocessing")
                            preprocessing_code_result = self.preprocessing_coder_agent(iteration_type=iteration_type)
                            if preprocessing_code_result.get("status") == "failed":
                                logger.error(f"Preprocessing failed for candidate {idx}: {preprocessing_code_result.get('error')}")
                                self.output_folder = str(original_output_folder)
                                continue
                            self.preprocessing_code = preprocessing_code_result.get("code")

                            # Modeling
                            if timeout_occurred.is_set():
                                raise IterationTimeoutError("Iteration timeout occurred before modeling")
                            modeling_code_result = self.modeling_coder_agent(iteration_type=iteration_type)
                            if modeling_code_result.get("status") == "failed":
                                logger.error(f"Modeling failed for candidate {idx}: {modeling_code_result.get('error')}")
                                self.output_folder = str(original_output_folder)
                                continue
                            self.modeling_code = modeling_code_result.get("code")

                            # Assembly
                            if timeout_occurred.is_set():
                                raise IterationTimeoutError("Iteration timeout occurred before assembly")
                            assembler_result = self.assembler_agent(iteration_type=iteration_type)
                            if assembler_result.get("status") == "failed":
                                logger.error(f"Assembly failed for candidate {idx}: {assembler_result.get('error')}")
                                self.output_folder = str(original_output_folder)
                                continue
                            candidate_success = True
                            self.assembled_code = assembler_result.get("code")
                            if not self._run_deployment_stage(iteration_type=iteration_type):
                                logger.error(f"Deployment stage failed for candidate {idx}.")
                                candidate_success = False

                        if candidate_success:
                            cand_submission = candidate_dir / "submission.csv"
                            if cand_submission.exists():
                                any_success = True
                                try:
                                    dst_archive = parent_iter_dir / f"submission_cand_{idx}.csv"
                                    shutil.copy2(cand_submission, dst_archive)
                                    if first_success_idx is None:
                                        shutil.copy2(cand_submission, parent_iter_dir / "submission.csv")
                                        first_success_idx = idx
                                except Exception as e:
                                    logger.warning(f"Could not copy submission for candidate {idx}: {e}")
                            else:
                                logger.error(f"Candidate {idx} reported success but produced no submission.csv; treating as failure.")

                        # Restore output folder for next candidate
                        self.output_folder = str(original_output_folder)

                    return any_success
            else:
                # Ensure other iterations are not influenced by retrieval results
                if hasattr(self, "model_suggestions"):
                    delattr(self, "model_suggestions")

            # Step 1c: Build iteration-specific knowledge pack (non-pretrained or no candidates)
            self._prepare_iteration_knowledge(iteration_type)

            # Step 1b: Run guideline agent
            if timeout_occurred.is_set():
                raise IterationTimeoutError("Iteration timeout occurred before guideline generation")
                
            if not self._prepare_guideline(iteration_type=iteration_type):
                return False

            if self.is_monolithic_mode():
                if timeout_occurred.is_set():
                    raise IterationTimeoutError("Iteration timeout occurred before monolithic generation")
                mono_result = self.monolithic_coder_agent(iteration_type=iteration_type)
                if mono_result.get("status") == "failed":
                    logger.error(f"Monolithic generation failed: {mono_result.get('error')}")
                    return False
                self.assembled_code = mono_result.get("code")
                if not self._run_deployment_stage(iteration_type=iteration_type):
                    return False
                return True

            # Step 2: Run Preprocessing Coder Agent
            if timeout_occurred.is_set():
                raise IterationTimeoutError("Iteration timeout occurred before preprocessing")
                
            preprocessing_code_result = self.preprocessing_coder_agent(iteration_type=iteration_type)
            if preprocessing_code_result.get("status") == "failed":
                logger.error(f"Preprocessing code generation failed: {preprocessing_code_result.get('error')}")
                return False
            self.preprocessing_code = preprocessing_code_result.get("code")
            logger.info("Preprocessing code generated and validated successfully.")

            # Step 3: Run Modeling Coder Agent
            if timeout_occurred.is_set():
                raise IterationTimeoutError("Iteration timeout occurred before modeling")
                
            modeling_code_result = self.modeling_coder_agent(iteration_type=iteration_type)
            if modeling_code_result.get("status") == "failed":
                logger.error(f"Modeling code generation failed: {modeling_code_result.get('error')}")
                return False
            self.modeling_code = modeling_code_result.get("code")
            logger.info("Modeling code generated successfully.")

            # Step 4: Run Assembler Agent
            if timeout_occurred.is_set():
                raise IterationTimeoutError("Iteration timeout occurred before assembly")
                
            assembler_result = self.assembler_agent(iteration_type=iteration_type)
            if assembler_result.get("status") == "failed":
                logger.error(f"Final code assembly and execution failed: {assembler_result.get('error')}")
                return False
            self.assembled_code = assembler_result.get("code")
            if not self._run_deployment_stage(iteration_type=iteration_type):
                return False
            logger.info("Final script generated and executed successfully.")
            
            return True
            
        except IterationTimeoutError:
            raise  # Re-raise timeout error
        except Exception as e:
            logger.error(f"Error in iteration pipeline: {e}")
            return False

    def run_pipeline_partial(self, stop_after="guideline"):
        """Run pipeline up to a specific checkpoint."""
        logger.info(f"Starting partial AutoML pipeline (stop after: {stop_after})...")

        # Step 1: Run description analysis agent
        analysis_result = self.description_analyzer_agent()
        if "error" in analysis_result:
            logger.error(f"Description analysis failed: {analysis_result['error']}")
            return False
        logger.info(f"Analysis result: {analysis_result}")
        self.description_analysis = analysis_result

        if stop_after == "description":
            logger.info("Pipeline stopped after description analysis.")
            return True

        # Step 2: Run profiling agent
        profiling_result = self.profiling_agent()
        if "error" in profiling_result:
            logger.error(f"Data profiling failed: {profiling_result['error']}")
            return False
        self.profiling_result = profiling_result
        logger.info("Profiling overview generated.")

        if stop_after == "profiling":
            logger.info("Pipeline stopped after profiling.")
            return True

        # Step 3a: Summarize profiling via LLM
        profiling_summary = self.profiling_summarizer_agent()
        if "error" in profiling_summary:
            logger.error(f"Profiling summarization failed: {profiling_summary['error']}")
            return False
        self.profiling_summary = profiling_summary

        # Step 3b: Task schema inference
        self._prepare_task_schema()

        # Step 3b: Retrieve pretrained model suggestions
        model_suggestions = self._run_planning_agent_with_retries(
            "ModelRetrieverAgent",
            lambda: self.model_retriever_agent(),
            max_retries=2,
        )
        self.model_suggestions = model_suggestions

        if stop_after == "pre-guideline":
            # Save the default prompt template for editing
            self.save_default_guideline_prompt_template()
            logger.info("Pipeline stopped before guideline generation.")
            logger.info("You can now edit the guideline prompt template and resume from guideline generation.")
            return True

        # Step 3c: Run guideline agent
        if not self._prepare_guideline():
            return False

        if stop_after == "guideline":
            logger.info("Pipeline stopped after guideline generation.")
            logger.info("You can now manually edit the guideline in the states folder.")
            return True

        logger.info("Partial AutoML pipeline completed successfully!")
        return True

    def load_checkpoint_state(self):
        """Load previously saved checkpoint state."""
        import json
        import os
        
        states_dir = os.path.join(self.output_folder, "states")
        
        # Debug: List all files in states directory
        if os.path.exists(states_dir):
            files_in_states = os.listdir(states_dir)
            logger.info(f"Files found in states directory: {files_in_states}")
        else:
            logger.warning(f"States directory does not exist: {states_dir}")
            return
        
        # Load description analysis
        desc_file = os.path.join(states_dir, "description_analyzer_response.json")
        if os.path.exists(desc_file):
            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    self.description_analysis = json.load(f)
                logger.info("Loaded description analysis from checkpoint")
            except Exception as e:
                logger.error(f"Failed to load description analysis: {e}")
        else:
            logger.warning(f"Description analysis file not found: {desc_file}")
            # Initialize as None so we can check later
            self.description_analysis = None
        
        # Load profiling result
        prof_file = os.path.join(states_dir, "profiling_result.json")
        if os.path.exists(prof_file):
            with open(prof_file, 'r', encoding='utf-8') as f:
                self.profiling_result = json.load(f)
            logger.info("Loaded profiling result from checkpoint")
        
        # Load profiling summary  
        prof_sum_file = os.path.join(states_dir, "profiling_summary.json")
        if os.path.exists(prof_sum_file):
            with open(prof_sum_file, 'r', encoding='utf-8') as f:
                self.profiling_summary = json.load(f)
            logger.info("Loaded profiling summary from checkpoint")
        
        # Load model suggestions
        model_file = os.path.join(states_dir, "model_retrieval.json")
        if os.path.exists(model_file):
            with open(model_file, 'r', encoding='utf-8') as f:
                self.model_suggestions = json.load(f)
            logger.info("Loaded model suggestions from checkpoint")

        # Load task schema and task context if available
        task_schema_file = os.path.join(states_dir, "task_schema.json")
        if os.path.exists(task_schema_file):
            try:
                with open(task_schema_file, 'r', encoding='utf-8') as f:
                    self.task_schema = json.load(f)
                logger.info("Loaded task schema from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load task schema: {e}")

        task_context_file = os.path.join(states_dir, "task_context.json")
        if os.path.exists(task_context_file):
            try:
                with open(task_context_file, 'r', encoding='utf-8') as f:
                    self.task_context = json.load(f)
                logger.info("Loaded task context from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load task context: {e}")
        
        # Load guideline (might be manually edited)
        guideline_file = os.path.join(states_dir, "guideline", "guideline_response.json")
        if not os.path.exists(guideline_file):
            # Backward compatibility
            legacy = os.path.join(states_dir, "guideline_response.json")
            guideline_file = legacy if os.path.exists(legacy) else guideline_file
        if os.path.exists(guideline_file):
            with open(guideline_file, 'r', encoding='utf-8') as f:
                self.guideline = json.load(f)
            logger.info("Loaded guideline from checkpoint")

    def resume_pipeline_from_checkpoint(self, start_from="preprocessing"):
        """Resume pipeline from a specific checkpoint."""
        logger.info(f"Resuming AutoML pipeline from: {start_from}...")
        
        # Load previous state
        self.load_checkpoint_state()
        
        # Validate required states are loaded
        if not hasattr(self, 'description_analysis') or self.description_analysis is None:
            logger.error("Cannot resume: description_analysis not found or is None")
            logger.error("Make sure you have run the pipeline at least until the description analysis step")
            return False
        
        # For resume from guideline, we don't need existing guideline
        if start_from != "guideline" and (not hasattr(self, 'guideline') or self.guideline is None):
            logger.error(f"Cannot resume from {start_from}: guideline not found")
            logger.error("For this resume point, you need to have run until guideline generation")
            return False

        if start_from == "guideline":
            # Load custom prompt template if available
            self.update_guideline_prompt_template()
            
            # Check if we have necessary data for guideline generation
            if not hasattr(self, 'profiling_result') or not hasattr(self, 'model_suggestions'):
                logger.warning("Missing profiling or model suggestions data. Running those steps first...")
                
                # Re-run profiling if needed
                if not hasattr(self, 'profiling_result'):
                    profiling_result = self.profiling_agent()
                    if "error" in profiling_result:
                        logger.error(f"Data profiling failed: {profiling_result['error']}")
                        return False
                    self.profiling_result = profiling_result
                
                # Re-run profiling summary if needed
                if not hasattr(self, 'profiling_summary'):
                    profiling_summary = self.profiling_summarizer_agent()
                    if "error" in profiling_summary:
                        logger.error(f"Profiling summarization failed: {profiling_summary['error']}")
                        return False
                    self.profiling_summary = profiling_summary

                # Re-run task schema if needed
                if not hasattr(self, 'task_context') or not self.task_context:
                    self._prepare_task_schema()
                
                # Re-run model retrieval if needed
                if not hasattr(self, 'model_suggestions'):
                    model_suggestions = self._run_planning_agent_with_retries(
                        "ModelRetrieverAgent",
                        lambda: self.model_retriever_agent(),
                        max_retries=2,
                    )
                    self.model_suggestions = model_suggestions
            
            # Re-run guideline generation (useful after editing prompt)
            if not self._prepare_guideline():
                return False
            logger.info("Guideline regenerated successfully.")

        if start_from in ["guideline", "preprocessing"]:
            # Step 4: Run Preprocessing Coder Agent
            preprocessing_code_result = self.preprocessing_coder_agent()
            if preprocessing_code_result.get("status") == "failed":
                logger.error(f"Preprocessing code generation failed: {preprocessing_code_result.get('error')}")
                return False
            self.preprocessing_code = preprocessing_code_result.get("code")
            logger.info("Preprocessing code generated and validated successfully.")

        if start_from in ["guideline", "preprocessing", "modeling"]:
            # Step 5: Run Modeling Coder Agent
            modeling_code_result = self.modeling_coder_agent()
            if modeling_code_result.get("status") == "failed":
                logger.error(f"Modeling code generation failed: {modeling_code_result.get('error')}")
                return False
            self.modeling_code = modeling_code_result.get("code")
            logger.info("Modeling code generated successfully.")

        if start_from in ["guideline", "preprocessing", "modeling", "assemble"]:
            # Step 6: Run Assembler Agent
            assembler_result = self.assembler_agent()
            if assembler_result.get("status") == "failed":
                logger.error(f"Final code assembly and execution failed: {assembler_result.get('error')}")
                return False
            self.assembled_code = assembler_result.get("code")
            if not self._run_deployment_stage(iteration_type="default"):
                return False
            logger.info("Initial script generated and executed successfully.")

        logger.info("AutoML pipeline completed successfully!")
        return True

    def update_guideline_prompt_template(self, new_template: str = None):
        """
        Update the guideline prompt template.
        If new_template is None, it will load from a file if it exists.
        """
        import os
        
        # Try to load from file first
        custom_prompt_file = os.path.join(self.output_folder, "custom_guideline_prompt.txt")
        if new_template is None and os.path.exists(custom_prompt_file):
            with open(custom_prompt_file, 'r', encoding='utf-8') as f:
                new_template = f.read()
            logger.info("Loaded custom guideline prompt from file")
        elif new_template is None:
            logger.info("No custom prompt template provided, using default")
            return
        
        # Update the template
        self.guideline_agent.prompt_handler.template = new_template
        logger.info("Guideline prompt template updated")
        
        # Save the template for reference
        self.save_and_log_states(new_template, "guideline_prompt_template_used.txt")

    def save_default_guideline_prompt_template(self):
        """Save the default guideline prompt template for editing."""
        default_template = self.guideline_agent.prompt_handler.default_template()
        template_file = os.path.join(self.output_folder, "custom_guideline_prompt.txt")
        
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(default_template)
        
        logger.info(f"Default guideline prompt template saved to: {template_file}")
        logger.info("You can edit this file and resume from guideline generation.")
        return template_file

    def run_pipeline_multi_iteration(self):
        """Run the pipeline with 3 iterations for different algorithm approaches."""
        logger.info("Starting Multi-Iteration AutoML Pipeline...")
        
        # Store original output folder
        original_output_folder = self.output_folder
        
        # Define iterations
        iterations = [
            {
                "name": "traditional",
                "folder": "iteration_1_traditional",
                "description": "Traditional ML algorithms (XGBoost, LightGBM, CatBoost)"
            },
            {
                "name": "custom_nn_search", 
                "folder": "iteration_2_custom_nn_search",
                "description": "Custom Neural Networks with Architecture Search"
            },
            {
                "name": "pretrained",
                "folder": "iteration_3_pretrained", 
                "description": "Pretrained Models"
            }
        ]

        monitor_cfg = self._get_hardware_monitoring_config()
        total_hw_monitor = None
        if monitor_cfg["enabled"]:
            try:
                total_hw_monitor = HardwareMonitor(
                    sample_interval_sec=monitor_cfg["sample_interval_sec"],
                    include_gpu=monitor_cfg["include_gpu"],
                    monitor_scope="host",
                    target_root_pid=os.getpid(),
                )
                total_hw_monitor.start(iteration_name="overall")
                logger.info(
                    f"Overall hardware monitor started (interval={monitor_cfg['sample_interval_sec']}s, include_gpu={monitor_cfg['include_gpu']})"
                )
            except Exception as e:
                total_hw_monitor = None
                logger.warning(f"Failed to start overall hardware monitor: {e}")
        
        # Run shared analysis steps once
        logger.info("Running shared analysis steps...")
        success = self._run_shared_analysis()
        if not success:
            if total_hw_monitor is not None:
                try:
                    total_hw_file = os.path.join(original_output_folder, "hardware_usage_total.json")
                    total_hw_monitor.stop_and_export(
                        output_json_path=total_hw_file,
                        run_id=Path(original_output_folder).name,
                        iteration_name="overall",
                        extra_metadata={
                            "execution_mode": "aborted_before_iterations",
                            "iteration_count": 0,
                        },
                    )
                except Exception:
                    pass
            return
            
        # Run each iteration (sequential or parallel)
        iteration_paths = []

        if self.parallel_iterations:
            logger.info("Parallel iteration mode enabled (process-based workers).")
            iteration_timeout_map: Dict[str, int] = {}
            for i, iteration in enumerate(iterations, 1):
                iteration_timeout = self.get_iteration_timeout(iteration['name'])
                iteration_timeout_map[iteration['name']] = iteration_timeout
                logger.info(f"=== Scheduling Iteration {i}: {iteration['description']} ===")
                logger.info(f"Timeout for {iteration['name']}: {iteration_timeout} seconds ({iteration_timeout/60:.1f} minutes)")
                iteration_output = os.path.join(original_output_folder, iteration['folder'])
                os.makedirs(iteration_output, exist_ok=True)
                iteration_paths.append(iteration_output)

            shared_context = {
                "description_analysis": getattr(self, "description_analysis", {}) or {},
                "profiling_result": getattr(self, "profiling_result", {}) or {},
                "profiling_summary": getattr(self, "profiling_summary", {}) or {},
                "task_schema": getattr(self, "task_schema", None),
                "task_context": getattr(self, "task_context", None),
            }
            config_dict = OmegaConf.to_container(self.config, resolve=True)

            worker_payloads: List[Dict[str, Any]] = []
            for iteration in iterations:
                worker_payloads.append(
                    {
                        "iteration": iteration,
                        "input_data_folder": self.input_data_folder,
                        "original_output_folder": original_output_folder,
                        "iteration_output": os.path.join(original_output_folder, iteration["folder"]),
                        "iteration_timeout": iteration_timeout_map[iteration["name"]],
                        "monitor_cfg": {**monitor_cfg, "monitor_scope": "process"},
                        "shared_context": shared_context,
                        "config_dict": config_dict,
                        "ablation_variant": self.ablation_variant,
                        "search_mode": self.search_mode,
                    }
                )

            max_workers = min(len(worker_payloads), int(getattr(self.config, "parallel_workers", len(worker_payloads)) or len(worker_payloads)))
            logger.info(f"Launching {len(worker_payloads)} iteration workers with max_workers={max_workers}.")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_name = {
                    executor.submit(_run_single_iteration_worker, payload): payload["iteration"]["name"]
                    for payload in worker_payloads
                }
                for future in as_completed(future_to_name):
                    iteration_name = future_to_name[future]
                    try:
                        worker_result = future.result()
                        logger.info(
                            f"Parallel iteration finished: {iteration_name} | status={worker_result.get('status')}"
                        )
                    except Exception as e:
                        logger.error(f"Parallel iteration worker crashed: {iteration_name} | error={e}")
        else:
            for i, iteration in enumerate(iterations, 1):
                iteration_timeout = self.get_iteration_timeout(iteration['name'])
                logger.info(f"=== Starting Iteration {i}: {iteration['description']} ===")
                logger.info(f"Timeout set for this iteration: {iteration_timeout} seconds ({iteration_timeout/60:.1f} minutes)")

                # Create iteration-specific output folder
                iteration_output = os.path.join(original_output_folder, iteration['folder'])
                os.makedirs(iteration_output, exist_ok=True)
                iteration_paths.append(iteration_output)

                # Temporarily change output folder for this iteration
                self.output_folder = iteration_output

                # Track iteration start time
                iteration_start_time = time.time()

                success = False
                iteration_status = "unknown"
                hw_monitor = None
                if monitor_cfg["enabled"]:
                    try:
                        hw_monitor = HardwareMonitor(
                            sample_interval_sec=monitor_cfg["sample_interval_sec"],
                            include_gpu=monitor_cfg["include_gpu"],
                            monitor_scope="process",
                            target_root_pid=os.getpid(),
                        )
                        hw_monitor.start(iteration_name=iteration["name"])
                        logger.info(
                            f"Hardware monitor started (interval={monitor_cfg['sample_interval_sec']}s, include_gpu={monitor_cfg['include_gpu']})"
                        )
                    except Exception as e:
                        hw_monitor = None
                        logger.warning(f"Failed to start hardware monitor: {e}")
                try:
                    # Run iteration with cross-platform timeout handling
                    success = self._run_iteration_with_timeout(iteration['name'], iteration_timeout)

                    # Calculate iteration duration
                    iteration_duration = time.time() - iteration_start_time

                    if success:
                        logger.info(f"=== Iteration {i} completed successfully in {iteration_duration:.1f} seconds ===")
                        iteration_status = "success"
                    else:
                        logger.error(f"Iteration {i} ({iteration['name']}) failed after {iteration_duration:.1f} seconds!")
                        iteration_status = "failed"

                except IterationTimeoutError:
                    iteration_duration = time.time() - iteration_start_time
                    logger.warning(f"⏰ Iteration {i} ({iteration['name']}) timed out after {iteration_timeout} seconds!")
                    logger.info(f"Moving to next iteration...")
                    success = False
                    iteration_status = "timeout"

                except Exception as e:
                    iteration_duration = time.time() - iteration_start_time
                    logger.error(f"Iteration {i} ({iteration['name']}) failed with error: {e}")
                    success = False
                    iteration_status = "error"
                finally:
                    if hw_monitor is not None:
                        try:
                            hw_file = os.path.join(iteration_output, "hardware_usage.json")
                            hw_monitor.stop_and_export(
                                output_json_path=hw_file,
                                run_id=Path(original_output_folder).name,
                                iteration_name=iteration["name"],
                                extra_metadata={
                                    "iteration_index": i,
                                    "iteration_folder": iteration["folder"],
                                    "iteration_description": iteration["description"],
                                    "result_status": iteration_status,
                                    "timeout_sec": iteration_timeout,
                                    "duration_sec": round(time.time() - iteration_start_time, 3),
                                    "execution_mode": "sequential",
                                },
                            )
                            logger.info(f"Hardware usage exported to: {hw_file}")
                        except Exception as e:
                            logger.warning(f"Failed to export hardware usage JSON: {e}")
        
        # Restore original output folder
        self.output_folder = original_output_folder
        
        # Extract comprehensive results from all iterations
        logger.info("=== Extracting results from all iterations ===")
        extractor = IterationResultExtractor()
        iteration_results = []
        
        for iteration_path in iteration_paths:
            result = extractor.extract_from_iteration_folder(iteration_path)
            iteration_results.append(result)
            logger.info(f"Extracted results from {result['iteration_name']}: {result['status']}")
        
        # Use LLM to intelligently compare and rank iterations
        logger.info("=== LLM-based Intelligent Iteration Comparison ===")
        comparison_result = self.comparison_agent(
            iteration_results=iteration_results,
            original_task_description=self.description_analysis
        )
        
        if "error" in comparison_result:
            logger.error(f"LLM comparison failed: {comparison_result['error']}")
            logger.info("Falling back to basic selection...")
            best_iteration_name = self._fallback_selection(iteration_results)
        else:
            best_iteration_name = comparison_result.get('best_iteration', {}).get('name')
            
            # Save detailed comparison report
            comparison_file = os.path.join(original_output_folder, "llm_comparison_results.json")
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_result, f, indent=2, ensure_ascii=False)
            logger.info(f"LLM comparison report saved to: {comparison_file}")
        
        # Copy best submission to final_submission folder
        if best_iteration_name:
            # Resolve folder by normalizing name (case/format-insensitive)
            def _normalize(s: str) -> str:
                import re
                return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") if s else ""

            candidates = []
            try:
                from pathlib import Path as _P
                root = _P(original_output_folder)
                for d in root.iterdir():
                    if d.is_dir() and d.name.startswith("iteration_"):
                        candidates.append(d)
            except Exception:
                pass

            resolved_path = None
            norm_target = _normalize(best_iteration_name)
            for d in candidates:
                if d.name == best_iteration_name or d.name.lower() == best_iteration_name.lower() or _normalize(d.name) == norm_target:
                    resolved_path = str(d)
                    break

            best_iteration_path = resolved_path or os.path.join(original_output_folder, best_iteration_name)
            success = self._copy_best_submission(best_iteration_path, original_output_folder)
            
            if success:
                logger.info(f"✅ Best submission copied from {best_iteration_name}")
                if "error" not in comparison_result:
                    logger.info(f"📊 LLM Reasoning: {comparison_result.get('reasoning_summary', 'No reasoning provided')}")
            else:
                logger.error("❌ Failed to copy best submission")
        else:
            logger.error("❌ No best iteration selected")
        
        # Print summary
        successful_count = len([r for r in iteration_results if r.get('status') == 'success'])
        logger.info(f"📈 Summary: {successful_count}/{len(iterations)} iterations successful")
        if best_iteration_name:
            logger.info(f"🏆 LLM Selected Winner: {best_iteration_name}")

        if total_hw_monitor is not None:
            try:
                total_hw_file = os.path.join(original_output_folder, "hardware_usage_total.json")
                total_hw_monitor.stop_and_export(
                    output_json_path=total_hw_file,
                    run_id=Path(original_output_folder).name,
                    iteration_name="overall",
                    extra_metadata={
                        "execution_mode": "parallel_process" if self.parallel_iterations else "sequential",
                        "iteration_count": len(iterations),
                    },
                )
                logger.info(f"Overall hardware usage exported to: {total_hw_file}")
            except Exception as e:
                logger.warning(f"Failed to export overall hardware usage JSON: {e}")

        # Best-effort: auto-generate hardware usage plot for this multi-iteration run.
        try:
            project_root = Path(__file__).resolve().parents[3]
            plot_script = project_root / "scripts" / "plot_hw_usage.py"
            if plot_script.exists():
                cmd = [
                    sys.executable,
                    str(plot_script),
                    "--run-dir",
                    str(original_output_folder),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode == 0:
                    logger.info(f"Hardware usage plot generated for run: {original_output_folder}")
                else:
                    stderr_preview = (proc.stderr or "").strip()[:500]
                    logger.warning(
                        f"Hardware usage auto-plot failed (code={proc.returncode}). stderr: {stderr_preview}"
                    )
            else:
                logger.warning(f"Hardware plot script not found: {plot_script}")
        except Exception as e:
            logger.warning(f"Failed to auto-generate hardware usage plot: {e}")
        
        logger.info("Multi-Iteration AutoML Pipeline completed!")
    
    def _fallback_selection(self, iteration_results: List[Dict]) -> str:
        """Fallback selection method when LLM comparison fails."""
        # Simple fallback: prefer successful iterations in priority order
        priority_order = ["iteration_1_pretrained", "iteration_2_traditional", "iteration_3_custom_nn"]

        successful_iterations = [
            r for r in iteration_results
            if r.get('status') == 'success'
        ]

        if not successful_iterations:
            logger.warning("No successful iterations found for fallback selection")
            return None

        # Select based on priority order
        for preferred_name in priority_order:
            for iteration in successful_iterations:
                if preferred_name in iteration.get('iteration_name', ''):
                    logger.info(f"Fallback selected: {iteration['iteration_name']}")
                    return iteration['iteration_name']

        # If no match, select first successful
        first_successful = successful_iterations[0]['iteration_name']
        logger.info(f"Fallback selected first successful: {first_successful}")
        return first_successful
    
    def _copy_best_submission(self, source_iteration_path: str, target_folder: str) -> bool:
        """Copy the best submission to final_submission folder."""
        try:
            source_path = Path(source_iteration_path)
            target_path = Path(target_folder) / "final_submission"
            
            # Create target directory
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Copy submission.csv
            source_submission = source_path / "submission.csv"
            target_submission = target_path / "submission.csv"
            
            if source_submission.exists():
                shutil.copy2(source_submission, target_submission)
                logger.info(f"Copied best submission from {source_submission} to {target_submission}")
                
                # Copy comparison metadata
                metadata = {
                    "source_iteration": source_path.name,
                    "copied_at": datetime.now().isoformat(),
                    "files_copied": ["submission.csv"]
                }
                metadata_file = target_path / "selection_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                return True
            else:
                logger.error(f"Source submission file not found: {source_submission}")
                return False
                
        except Exception as e:
            logger.error(f"Error copying best submission: {e}")
            return False

    def run_pipeline_single_iteration(self, iteration_type: str):
        """Run the pipeline with a single specific iteration approach."""
        logger.info(f"Starting Single-Iteration AutoML Pipeline ({iteration_type})...")
        
        # Run shared analysis steps once
        logger.info("Running shared analysis steps...")
        success = self._run_shared_analysis()
        if not success:
            return
        
        # Create iteration-specific output folder
        iteration_info = {
            "traditional": {"folder": "iteration_traditional", "description": "Traditional ML algorithms"},
            "custom_nn": {"folder": "iteration_custom_nn", "description": "Custom Neural Networks (no search)"}, 
            "custom_nn_search": {"folder": "iteration_custom_nn_search", "description": "Custom Neural Networks with Architecture Search"},
            "pretrained": {"folder": "iteration_pretrained", "description": "Pretrained Models"}
        }
        
        info = iteration_info.get(iteration_type, {"folder": f"iteration_{iteration_type}", "description": iteration_type})
        iteration_output = os.path.join(self.output_folder, info['folder'])
        os.makedirs(iteration_output, exist_ok=True)
        
        # Store original and temporarily change output folder
        original_output_folder = self.output_folder
        self.output_folder = iteration_output
        
        # Get iteration timeout
        iteration_timeout = self.get_iteration_timeout(iteration_type)
        logger.info(f"=== Running {info['description']} ===")
        logger.info(f"Timeout set for this iteration: {iteration_timeout} seconds ({iteration_timeout/60:.1f} minutes)")
        
        # Track iteration start time
        iteration_start_time = time.time()
        monitor_cfg = self._get_hardware_monitoring_config()
        hw_monitor = None
        if monitor_cfg["enabled"]:
            try:
                hw_monitor = HardwareMonitor(
                    sample_interval_sec=monitor_cfg["sample_interval_sec"],
                    include_gpu=monitor_cfg["include_gpu"],
                    monitor_scope="process",
                    target_root_pid=os.getpid(),
                )
                hw_monitor.start(iteration_name=iteration_type)
                logger.info(
                    f"Hardware monitor started (interval={monitor_cfg['sample_interval_sec']}s, include_gpu={monitor_cfg['include_gpu']})"
                )
            except Exception as e:
                hw_monitor = None
                logger.warning(f"Failed to start hardware monitor: {e}")
        
        success = False
        iteration_status = "unknown"
        try:
            # Run iteration with cross-platform timeout handling
            success = self._run_iteration_with_timeout(iteration_type, iteration_timeout)
            
            iteration_duration = time.time() - iteration_start_time
            if success:
                logger.info(f"Single-iteration pipeline ({iteration_type}) completed successfully in {iteration_duration:.1f} seconds!")
                iteration_status = "success"
            else:
                logger.error(f"Single-iteration pipeline ({iteration_type}) failed after {iteration_duration:.1f} seconds!")
                iteration_status = "failed"
                
        except IterationTimeoutError:
            iteration_duration = time.time() - iteration_start_time
            logger.warning(f"⏰ Single iteration ({iteration_type}) timed out after {iteration_timeout} seconds!")
            success = False
            iteration_status = "timeout"
            
        except Exception as e:
            iteration_duration = time.time() - iteration_start_time
            logger.error(f"Single iteration ({iteration_type}) failed with error: {e}")
            success = False
            iteration_status = "error"
            
        finally:
            if hw_monitor is not None:
                try:
                    hw_file = os.path.join(iteration_output, "hardware_usage.json")
                    hw_monitor.stop_and_export(
                        output_json_path=hw_file,
                        run_id=Path(original_output_folder).name,
                        iteration_name=iteration_type,
                        extra_metadata={
                            "iteration_folder": info["folder"],
                            "iteration_description": info["description"],
                            "result_status": iteration_status,
                            "timeout_sec": iteration_timeout,
                            "duration_sec": round(time.time() - iteration_start_time, 3),
                        },
                    )
                    logger.info(f"Hardware usage exported to: {hw_file}")
                except Exception as e:
                    logger.warning(f"Failed to export hardware usage JSON: {e}")
            # Restore original output folder
            self.output_folder = original_output_folder

    def run_pipeline_ablation(self, variant: str, iteration_type: str):
        """Run the pipeline under an ablation variant."""
        logger.info(f"=== Starting Ablation Run: {variant} | Iteration: {iteration_type} ===")
        prev_variant = self.ablation_variant
        self.ablation_variant = variant

        # Shared analysis
        logger.info("Running shared analysis steps...")
        success = self._run_shared_analysis()
        if not success:
            self.ablation_variant = prev_variant
            return

        ablation_root = Path(self.output_folder) / f"ablation_{variant}"
        ablation_root.mkdir(parents=True, exist_ok=True)
        iteration_folder = ablation_root / f"iteration_{iteration_type}"
        iteration_folder.mkdir(parents=True, exist_ok=True)

        original_output_folder = self.output_folder
        self.output_folder = str(iteration_folder)

        iteration_timeout = self.get_iteration_timeout(iteration_type)
        logger.info(f"Ablation iteration timeout: {iteration_timeout} seconds")

        start_time = time.time()
        try:
            success = self._run_iteration_with_timeout(iteration_type, iteration_timeout)
            duration = time.time() - start_time
            if success:
                logger.info(f"Ablation run ({variant}) completed successfully in {duration:.1f}s")
            else:
                logger.error(f"Ablation run ({variant}) failed after {duration:.1f}s")
        except IterationTimeoutError:
            duration = time.time() - start_time
            logger.warning(f"Ablation run ({variant}) timed out after {iteration_timeout} seconds.")
        finally:
            self.output_folder = original_output_folder
            self.ablation_variant = prev_variant
    
    def _run_shared_analysis(self):
        """Run the shared analysis steps (description, profiling, summarization)."""
        # Step 1: Run description analysis agent
        analysis_result = self.description_analyzer_agent()
        if "error" in analysis_result:
            logger.error(f"Description analysis failed: {analysis_result['error']}")
            return False
        logger.info(f"Analysis result: {analysis_result}")
        self.description_analysis = analysis_result

        # Step 2: Run profiling agent
        profiling_result = self.profiling_agent()
        if "error" in profiling_result:
            logger.error(f"Data profiling failed: {profiling_result['error']}")
            return False
        self.profiling_result = profiling_result
        logger.info("Profiling overview generated.")

        # Step 3a: Summarize profiling via LLM to reduce noise
        profiling_summary = self.profiling_summarizer_agent()
        if "error" in profiling_summary:
            logger.error(f"Profiling summarization failed: {profiling_summary['error']}")
            return False
        self.profiling_summary = profiling_summary

        # Step 3b: Task schema inference (LLM-based)
        self._prepare_task_schema()

    # Note: Model retrieval will be run only within the pretrained iteration.
        
        return True

    def _prepare_task_schema(self) -> bool:
        """Infer task schema and build task_context for downstream phases."""
        try:
            task_schema = self._run_planning_agent_with_retries(
                "TaskSchemaAgent",
                lambda: self.task_schema_agent(),
                max_retries=2,
            )
            if task_schema and "error" not in task_schema:
                self.task_schema = task_schema
                self.task_context = {
                    "description_analysis": getattr(self, "description_analysis", {}) or {},
                    "task_schema": task_schema,
                    "profiling_summary": getattr(self, "profiling_summary", {}) or {},
                }
                try:
                    self.save_and_log_states(
                        json.dumps(self.task_context, indent=2, ensure_ascii=False),
                        "task_context.json",
                    )
                except Exception:
                    pass
                return True
            logger.warning("TaskSchemaAgent returned an error or empty schema; continuing without task_context.")
        except Exception as e:
            logger.warning(f"TaskSchemaAgent failed: {e}")
        return False

    def _prepare_iteration_knowledge(self, iteration_type: str) -> None:
        """Build iteration-specific knowledge pack (no code) for downstream prompts."""
        model_suggestions = getattr(self, "model_suggestions", None) if iteration_type == "pretrained" else None
        architecture_suggestions = getattr(self, "architecture_suggestions", None) if iteration_type == "custom_nn_search" else None
        try:
            knowledge_pack = self._run_planning_agent_with_retries(
                "KnowledgeRetrievalAgent",
                lambda: self.knowledge_retrieval_agent(
                    iteration_type=iteration_type,
                    model_suggestions=model_suggestions,
                    architecture_suggestions=architecture_suggestions,
                ),
                max_retries=2,
            )
            if knowledge_pack and "error" not in knowledge_pack:
                self.knowledge_packs[iteration_type or "default"] = knowledge_pack
            else:
                logger.warning("KnowledgeRetrievalAgent returned empty/errored pack; continuing without it.")
        except Exception as e:
            logger.warning(f"KnowledgeRetrievalAgent failed: {e}")

    def build_debug_context(self, stderr: str, code: str, phase_name: str, attempt: int) -> str:
        """
        Legacy/simple debug context (no triage/evidence stage).
        Keep behavior close to earlier pipeline: pass description-only context.
        """
        return (self.description_analysis or {}).get("task_description") or json.dumps(self.description_analysis or {})
    
    def _run_iteration_pipeline(self, iteration_type):
        """Run the pipeline for a specific iteration type."""
        # Step 1a: For custom_nn_search iteration, run architecture search
        if iteration_type == "custom_nn_search":
            architecture_suggestions = self._run_planning_agent_with_retries(
                "ArchitectureRetrieverAgent",
                lambda: self.architecture_retriever_agent(),
                max_retries=2,
            )
            if architecture_suggestions and "error" not in architecture_suggestions and architecture_suggestions.get("architectures"):
                self.architecture_suggestions = architecture_suggestions
                logger.info("Architecture search completed for custom_nn_search iteration.")
            else:
                # Fallback: continue without search (LLM designs from scratch)
                logger.warning("Architecture search failed, LLM will design from scratch.")
                self.architecture_suggestions = None
        else:
            # Ensure other iterations (including custom_nn) are not influenced by architecture search
            if hasattr(self, "architecture_suggestions"):
                delattr(self, "architecture_suggestions")
        
        # Step 1b: For pretrained iteration, per-candidate; otherwise normal flow
        if iteration_type == "pretrained":
            model_suggestions = self._run_planning_agent_with_retries(
                "ModelRetrieverAgent",
                lambda: self.model_retriever_agent(),
                max_retries=2,
            )
            
            # Check for critical errors (not just empty results)
            if not model_suggestions:
                logger.error("Pretrained iteration aborted: model_retriever_agent returned None.")
                return False
            if "error" in model_suggestions:
                logger.error(f"Pretrained iteration aborted: {model_suggestions.get('message', 'Unknown error')}")
                return False
                
            self.model_suggestions = model_suggestions
            models = self.model_suggestions.get("sota_models", [])
            
            # If no valid candidates found, fall back to LLM choosing model itself
            if not models:
                logger.warning("No valid SOTA candidates found. Falling back to single guideline generation (LLM will choose model).")
                # Continue with standard flow (non-candidate loop) - skip to line 1039
            else:
                # Build iteration-specific knowledge pack before candidate loop
                self._prepare_iteration_knowledge(iteration_type)
                # Candidate-wise loop: for each SOTA model, run guideline -> preprocessing -> modeling -> assembly
                any_success = False
                parent_iter_dir = Path(self.output_folder)
                first_success_idx = None
                for idx, cand in enumerate(models, start=1):
                    candidate_success = False
                    # Narrow suggestions to a single candidate
                    self.model_suggestions = {"sota_models": [cand], "source": model_suggestions.get("source", "sota-search")}
                    # Prepare candidate-specific output directory and switch context
                    candidate_dir = parent_iter_dir / f"candidate_{idx}"
                    candidate_dir.mkdir(parents=True, exist_ok=True)
                    original_output_folder = self.output_folder
                    self.output_folder = str(candidate_dir)
                    # Guideline
                    if not self._prepare_guideline(iteration_type=iteration_type):
                        logger.error(f"Guideline generation failed for candidate {idx}.")
                        self.output_folder = str(original_output_folder)
                        continue

                    if self.is_monolithic_mode():
                        mono_result = self.monolithic_coder_agent(iteration_type=iteration_type)
                        if mono_result.get("status") == "failed":
                            logger.error(f"Monolithic generation failed for candidate {idx}: {mono_result.get('error')}")
                            self.output_folder = str(original_output_folder)
                            continue
                        candidate_success = True
                        self.assembled_code = mono_result.get("code")
                        if not self._run_deployment_stage(iteration_type=iteration_type):
                            logger.error(f"Deployment stage failed for candidate {idx}.")
                            candidate_success = False
                    else:
                        # Preprocessing
                        preprocessing_code_result = self.preprocessing_coder_agent(iteration_type=iteration_type)
                        if preprocessing_code_result.get("status") == "failed":
                            logger.error(f"Preprocessing failed for candidate {idx}: {preprocessing_code_result.get('error')}")
                            self.output_folder = str(original_output_folder)
                            continue
                        self.preprocessing_code = preprocessing_code_result.get("code")

                        # Modeling
                        modeling_code_result = self.modeling_coder_agent(iteration_type=iteration_type)
                        if modeling_code_result.get("status") == "failed":
                            logger.error(f"Modeling failed for candidate {idx}: {modeling_code_result.get('error')}")
                            self.output_folder = str(original_output_folder)
                            continue
                        self.modeling_code = modeling_code_result.get("code")

                        # Assembly
                        assembler_result = self.assembler_agent(iteration_type=iteration_type)
                        if assembler_result.get("status") == "failed":
                            logger.error(f"Assembly failed for candidate {idx}: {assembler_result.get('error')}")
                            self.output_folder = str(original_output_folder)
                            continue
                        candidate_success = True
                        self.assembled_code = assembler_result.get("code")
                        if not self._run_deployment_stage(iteration_type=iteration_type):
                            logger.error(f"Deployment stage failed for candidate {idx}.")
                            candidate_success = False

                    if candidate_success:
                        cand_submission = candidate_dir / "submission.csv"
                        if cand_submission.exists():
                            any_success = True
                            try:
                                dst_archive = parent_iter_dir / f"submission_cand_{idx}.csv"
                                shutil.copy2(cand_submission, dst_archive)
                                if first_success_idx is None:
                                    shutil.copy2(cand_submission, parent_iter_dir / "submission.csv")
                                    first_success_idx = idx
                            except Exception as e:
                                logger.warning(f"Could not copy submission for candidate {idx}: {e}")
                        else:
                            logger.error(f"Candidate {idx} reported success but produced no submission.csv; treating as failure.")

                    # Restore output folder for next candidate
                    self.output_folder = str(original_output_folder)

                return any_success

        else:
            if hasattr(self, "model_suggestions"):
                delattr(self, "model_suggestions")

        # Step 1c: Build iteration-specific knowledge pack (non-pretrained or no candidates)
        self._prepare_iteration_knowledge(iteration_type)

        # Step 1b: Run guideline agent with iteration-specific algorithm constraint
        if not self._prepare_guideline(iteration_type=iteration_type):
            return False

        if self.is_monolithic_mode():
            mono_result = self.monolithic_coder_agent(iteration_type=iteration_type)
            if mono_result.get("status") == "failed":
                logger.error(f"Monolithic generation failed: {mono_result.get('error')}")
                return False
            self.assembled_code = mono_result.get("code")
            if not self._run_deployment_stage(iteration_type=iteration_type):
                return False
            return True

        # Step 2: Run Preprocessing Coder Agent
        preprocessing_code_result = self.preprocessing_coder_agent(iteration_type=iteration_type)
        if preprocessing_code_result.get("status") == "failed":
            logger.error(f"Preprocessing code generation failed: {preprocessing_code_result.get('error')}")
            return False
        self.preprocessing_code = preprocessing_code_result.get("code")
        logger.info("Preprocessing code generated and validated successfully.")

        # Step 3: Run Modeling Coder Agent
        modeling_code_result = self.modeling_coder_agent(iteration_type=iteration_type)
        if modeling_code_result.get("status") == "failed":
            logger.error(f"Modeling code generation failed: {modeling_code_result.get('error')}")
            return False
        self.modeling_code = modeling_code_result.get("code")
        logger.info("Modeling code generated successfully.")

        # Step 4: Run Assembler Agent
        assembler_result = self.assembler_agent(iteration_type=iteration_type)
        if assembler_result.get("status") == "failed":
            logger.error(f"Final code assembly and execution failed: {assembler_result.get('error')}")
            return False
        self.assembled_code = assembler_result.get("code")
        if not self._run_deployment_stage(iteration_type=iteration_type):
            return False
        logger.info("Final script generated and executed successfully.")
        
        return True

    def run_pipeline(self):
        """Run the entire pipeline from description analysis to code generation."""

        # Step 1: Run description analysis agent
        analysis_result = self.description_analyzer_agent()
        if "error" in analysis_result:
            logger.error(f"Description analysis failed: {analysis_result['error']}")
            return
        logger.info(f"Analysis result: {analysis_result}")

        self.description_analysis = analysis_result

        # Step 2: Run profiling agent
        profiling_result = self.profiling_agent()
        if "error" in profiling_result:
            logger.error(f"Data profiling failed: {profiling_result['error']}")
            return
        
        self.profiling_result = profiling_result
        logger.info("Profiling overview generated.")

        # Step 3: Run guideline agent
        # 3a: Summarize profiling via LLM to reduce noise
        profiling_summary = self.profiling_summarizer_agent()
        if "error" in profiling_summary:
            logger.error(f"Profiling summarization failed: {profiling_summary['error']}")
            return
        self.profiling_summary = profiling_summary

        # 3b: Task schema inference
        self._prepare_task_schema()

        # 3b: Retrieve pretrained model/embedding suggestions
        model_suggestions = self._run_planning_agent_with_retries(
            "ModelRetrieverAgent",
            lambda: self.model_retriever_agent(),
            max_retries=2,
        )
        self.model_suggestions = model_suggestions

        # 3c: Build knowledge pack for default iteration
        self._prepare_iteration_knowledge("default")

        # 3c: Run guideline agent with summarized profiling + model suggestions
        if not self._prepare_guideline():
            return

        # Step 4: Run Preprocessing Coder Agent
        preprocessing_code_result = self.preprocessing_coder_agent()
        if preprocessing_code_result.get("status") == "failed":
            logger.error(f"Preprocessing code generation failed: {preprocessing_code_result.get('error')}")
            return

        self.preprocessing_code = preprocessing_code_result.get("code")
        logger.info("Preprocessing code generated and validated successfully.")

        # Step 5: Run Modeling Coder Agent
        modeling_code_result = self.modeling_coder_agent()
        if modeling_code_result.get("status") == "failed":
            logger.error(f"Modeling code generation failed: {modeling_code_result.get('error')}")
            return
            
        self.modeling_code = modeling_code_result.get("code")
        logger.info("Modeling code generated successfully (not yet validated).")

        # Step 6: Run Assembler Agent to assemble, finalize, and run the code
        assembler_result = self.assembler_agent()
        if assembler_result.get("status") == "failed":
            logger.error(f"Final code assembly and execution failed: {assembler_result.get('error')}")
            return
        
        self.assembled_code = assembler_result.get("code")
        if not self._run_deployment_stage(iteration_type="default"):
            return
        logger.info(f"Initial script generated and executed successfully.")

        logger.info("AutoML pipeline completed successfully!")

    def write_code_script(self, script, output_code_file):
        with open(output_code_file, "w") as file:
            file.write(script)

    def execute_code(self, code_to_execute: str, phase_name: str, attempt: int) -> dict:
        """
        Executes a string of Python code in a subprocess and saves the script,
        stdout, and stderr to a structured attempts folder.

        Args:
            code_to_execute: The Python code to run.
            phase_name: The name of the phase (e.g., "preprocessing", "assemble").
            attempt: The retry attempt number.

        Returns:
            A dictionary with execution status, stdout, and stderr.
        """
        # Create a structured directory for this attempt (under states)
        attempt_dir = Path(self.output_folder) / "states" / phase_name / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        # Define file paths for the script, stdout, and stderr
        script_path = attempt_dir / "generated_code.py"
        stdout_path = attempt_dir / "stdout.txt"
        stderr_path = attempt_dir / "stderr.txt"

        # Write the code to the script file
        self.write_code_script(code_to_execute, str(script_path))

        # In static ablation mode, skip execution for intermediate phases
        skip_execution = self.is_static_mode() and phase_name not in {"assemble"}
        if skip_execution:
            logger.info(f"[iMLstatic] Skipping runtime execution for phase '{phase_name}'. Performing syntax check only.")
            try:
                compile(code_to_execute, str(script_path), "exec")
                return {"success": True, "stdout": "", "stderr": ""}
            except SyntaxError as exc:
                return {"success": False, "stdout": "", "stderr": f"SyntaxError: {exc}"}

        logger.info(f"Executing code from: {script_path}")

        try:
            # Execute the script using subprocess with live streaming
            import select
            # Run from the dataset root so relative paths from description analyzer resolve
            working_dir = str(Path(self.input_data_folder).parent)

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=working_dir,
                env=build_child_execution_env(),
            )

            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            streams = [process.stdout, process.stderr]
            start_time = time.time()
            timeout = self.config.per_execution_timeout

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

            # Ensure process exits
            if process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stderr_chunks.append("Process forcibly terminated after timeout\n")

            stdout = "".join(stdout_chunks)
            stderr = "".join(stderr_chunks)

            # Save outputs
            with open(stdout_path, "w") as f:
                f.write(stdout)
            with open(stderr_path, "w") as f:
                f.write(stderr)

            if process.returncode == 0:
                logger.info("Code executed successfully.")
                return {"success": True, "stdout": stdout, "stderr": stderr}
            else:
                logger.error(f"Code execution failed with return code {process.returncode}.")
                full_error = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                return {"success": False, "stdout": stdout, "stderr": full_error}
        except Exception as e:
            logger.error(f"An exception occurred during code execution: {e}")
            with open(stderr_path, "w") as f:
                f.write(str(e))
            return {"success": False, "stdout": "", "stderr": str(e)}


    def update_python_code(self):
        """Update the current Python code."""
        assert len(self.python_codes) == self.time_step
        assert len(self.python_file_paths) == self.time_step

        python_code = self.python_coder()

        python_file_path = os.path.join(self.iteration_folder, "generated_code.py")

        self.write_code_script(python_code, python_file_path)

        self.python_codes.append(python_code)
        self.python_file_paths.append(python_file_path)

    def update_bash_script(self):
        """Update the current bash script."""
        assert len(self.bash_scripts) == self.time_step

        bash_script = self.bash_coder()

        bash_file_path = os.path.join(self.iteration_folder, "execution_script.sh")

        self.write_code_script(bash_script, bash_file_path)

        self.bash_scripts.append(bash_script)

    def execute_code_old(self):
        planner_decision, planner_error_summary, planner_prompt, stderr, stdout = self.executer(
            code_to_execute=self.bash_script,
            code_to_analyze=self.python_code,
            task_description=self.task_description,
            data_prompt=self.data_prompt,
        )

        self.save_and_log_states(stderr, "stderr", add_uuid=False)
        self.save_and_log_states(stdout, "stdout", add_uuid=False)

        if planner_decision == "FIX":
            logger.brief(f"[bold red]Code generation failed in iteration[/bold red] {self.time_step}!")
            # Add suggestions to the error message to guide next iteration
            error_message = f"stderr: {stderr}\n\n" if stderr else ""
            error_message += (
                f"Error summary from planner (the error can appear in stdout if it's catched): {planner_error_summary}"
            )
            self.update_error_message(error_message=error_message)
            return False
        elif planner_decision == "FINISH":
            logger.brief(
                f"[bold green]Code generation successful after[/bold green] {self.time_step + 1} [bold green]iterations[/bold green]"
            )
            self.update_error_message(error_message="")
            return True
        else:
            logger.warning(f"###INVALID Planner Output: {planner_decision}###")
            self.update_error_message(error_message="")
            return False

    def update_error_message(self, error_message: str):
        """Update the current error message."""
        assert len(self.error_messages) == self.time_step
        self.error_messages.append(error_message)

    def save_and_log_states(self, content, save_name, add_uuid=False):
        """
        Save content under output_folder/states. save_name can include nested folders
        like "guideline/guideline_prompt.txt".

        - When add_uuid is True, append a short UUID before the file extension.
        - Content may be a list or string; None is saved as "<None>".
        """
        # Optionally add a short UUID suffix to the filename
        if add_uuid:
            name, ext = os.path.splitext(save_name)
            uuid_suffix = str(uuid.uuid4()).replace("-", "")[:4]
            save_name = f"{name}_{uuid_suffix}{ext}"

        # Compose full path and ensure parent directories exist
        states_dir = os.path.join(self.output_folder, "states")
        output_file = os.path.join(states_dir, save_name)
        parent_dir = os.path.dirname(output_file)
        os.makedirs(parent_dir, exist_ok=True)

        logger.info(f"Saving {output_file}...")
        with open(output_file, "w") as file:
            if content is None:
                file.write("<None>")
            elif isinstance(content, list):
                file.write("\n".join(str(item) for item in content))
            else:
                file.write(content)

    def log_agent_start(self, message: str):
        logger.brief(message)

    def log_agent_end(self, message: str):
        logger.brief(message)

    def report_token_usage(self):
        token_usage_path = os.path.join(self.output_folder, "token_usage.json")
        usage = ChatLLMFactory.get_total_token_usage(save_path=token_usage_path)
        total = usage["total"]
        logger.brief(
            f"Total tokens — input: {total['total_input_tokens']}, "
            f"output: {total['total_output_tokens']}, "
            f"sum: {total['total_tokens']}"
        )

        logger.info(f"Full token usage detail:\n{usage}")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "retriever"):
            self.retriever.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
