from __future__ import annotations

import json
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Optional

from .main_runner import run_automl_pipeline


@dataclass
class DeploymentInfo:
    """Lightweight info about a resolved deployment package."""

    deployment_dir: Path
    manifest_path: Optional[Path]
    metadata_path: Optional[Path]
    deployment_py_path: Optional[Path]
    iteration_name: str


class IMLPredictor:
    """
    Public API template for using iML like a library (AutoGluon-style).

    Intended usage:
        predictor = IMLPredictor(path="./artifacts/run_001")
        predictor.fit(input_data_folder="./input/my_dataset")

        loaded = IMLPredictor.load("./artifacts/run_001")
        preds = loaded.predict(df_or_payload)

    Notes:
    - `fit()` orchestrates the existing iML pipeline.
    - `load()` resolves deployment artifacts from either a deployment folder or a run folder.
    - `predict()` delegates to a pluggable inference function. This is a template entrypoint,
      so plug in your task-specific inference logic to avoid coupling to one artifact format.
    """

    def __init__(
        self,
        *,
        label: Optional[str] = None,
        path: str | Path = "./runs",
        config_path: str = "configs/default.yaml",
        inference_fn: Optional[Callable[[Any, DeploymentInfo], Any]] = None,
    ) -> None:
        self.label = label
        self.path = Path(path)
        self.config_path = config_path
        self.inference_fn = inference_fn

        self.run_dir: Optional[Path] = None
        self.deployment: Optional[DeploymentInfo] = None
        self.deployments: Dict[str, DeploymentInfo] = {}
        self.active_iteration: Optional[str] = None
        self._module_cache: Dict[str, ModuleType] = {}

    def fit(
        self,
        *,
        input_data_folder: str,
        checkpoint_mode: str = "full",
        checkpoint_action: str = "guideline",
        single_iteration: Optional[str] = None,
        ablation_variant: Optional[str] = None,
        search_mode: Optional[str] = None,
        parallel_iterations: bool = True,
        output_folder: Optional[str | Path] = None,
    ) -> "IMLPredictor":
        """
        Run iML pipeline and keep path references for later `load()` / `predict()`.

        Args:
            parallel_iterations:
                Default is True (parallel). Set False to force sequential iteration
                execution, e.g. for lower resource usage or easier debugging.

        Returns:
            self
        """
        target_output = Path(output_folder) if output_folder else self.path
        target_output.mkdir(parents=True, exist_ok=True)

        run_automl_pipeline(
            input_data_folder=input_data_folder,
            output_folder=str(target_output),
            config_path=self.config_path,
            checkpoint_mode=checkpoint_mode,
            checkpoint_action=checkpoint_action,
            single_iteration=single_iteration,
            ablation_variant=ablation_variant,
            search_mode=search_mode,
            parallel_iterations=parallel_iterations,
        )

        self.run_dir = target_output
        self.deployments = self._discover_deployments(target_output)
        if not self.deployments:
            raise FileNotFoundError("No deployable iteration found (deployment/deployment.py missing).")

        self.active_iteration = self._pick_default_iteration(target_output, self.deployments)
        self.deployment = self.deployments[self.active_iteration]
        return self

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        iteration: Optional[str] = None,
        inference_fn: Optional[Callable[[Any, DeploymentInfo], Any]] = None,
    ) -> "IMLPredictor":
        """
        Load predictor from an existing iML run folder or deployment folder.
        """
        instance = cls(path=path, inference_fn=inference_fn)
        p = Path(path)

        if p.is_dir() and (p / "manifest.json").exists():
            # Direct deployment folder
            instance.run_dir = p.parent
            info = DeploymentInfo(
                deployment_dir=p,
                manifest_path=p / "manifest.json",
                metadata_path=(p / "metadata.json") if (p / "metadata.json").exists() else None,
                deployment_py_path=(p / "deployment.py") if (p / "deployment.py").exists() else None,
                iteration_name="direct",
            )
            instance.deployments = {"direct": info}
            instance.active_iteration = "direct"
            instance.deployment = info
            return instance

        instance.run_dir = p
        instance.deployments = instance._discover_deployments(p)
        if not instance.deployments:
            raise FileNotFoundError("No deployable iteration found (deployment/deployment.py missing).")

        selected = iteration or instance._pick_default_iteration(p, instance.deployments)
        if selected not in instance.deployments:
            raise ValueError(f"Iteration '{selected}' not found. Available: {list(instance.deployments.keys())}")

        instance.active_iteration = selected
        instance.deployment = instance.deployments[selected]
        return instance

    def list_iterations(self) -> list[str]:
        """Return available deployable iterations."""
        return sorted(self.deployments.keys())

    def use_iteration(self, iteration: str) -> "IMLPredictor":
        """Switch active iteration for subsequent predict() calls."""
        if iteration not in self.deployments:
            raise ValueError(f"Iteration '{iteration}' not found. Available: {list(self.deployments.keys())}")
        self.active_iteration = iteration
        self.deployment = self.deployments[iteration]
        return self

    def predict(self, data: Any) -> Any:
        """
        Run inference using deployment artifacts.

        Output contract (enforced by DeploymentAgent prompt):
            - Classification: list of dicts, one per input sample, in input order:
                  [{"label": <decoded_label>, "confidence": <float in [0, 1]>}, ...]
            - Regression: list of dicts, one per input sample:
                  [{"value": <float>}, ...]
            - Single-sample inputs still return a list of length 1.

        This method is intentionally adapter-based as a template:
        plug in a project-specific `inference_fn` if you need a different shape.
        Custom `inference_fn` overrides take precedence and are NOT required to
        follow the contract above.
        """
        if self.deployment is None:
            raise RuntimeError("Predictor is not loaded. Call fit(...) or load(...).")

        if self.inference_fn is None:
            return self._predict_via_deployment_module(data, self.deployment)

        return self.inference_fn(data, self.deployment)

    def predict_from_file(self, input_path: str | Path) -> Any:
        """
        Convenience template: load JSON/CSV payload then call `predict()`.
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            # Keep generic to avoid forcing pandas dependency in API template.
            payload = path.read_text(encoding="utf-8")

        return self.predict(payload)

    # ------------------------
    # Internal helpers
    # ------------------------
    def _discover_deployments(self, base_path: Path) -> Dict[str, DeploymentInfo]:
        """Discover deployable iterations (must have deployment/deployment.py)."""
        deployments: Dict[str, DeploymentInfo] = {}

        def _add(iteration_name: str, dep_dir: Path) -> None:
            dep_py = dep_dir / "deployment.py"
            if not dep_py.exists():
                return
            manifest = dep_dir / "manifest.json"
            metadata = dep_dir / "metadata.json"
            deployments[iteration_name] = DeploymentInfo(
                deployment_dir=dep_dir,
                manifest_path=manifest if manifest.exists() else None,
                metadata_path=metadata if metadata.exists() else None,
                deployment_py_path=dep_py,
                iteration_name=iteration_name,
            )

        if (base_path / "deployment").is_dir():
            _add("default", base_path / "deployment")

        for d in sorted(base_path.glob("iteration_*")):
            if not d.is_dir():
                continue
            dep = d / "deployment"
            if dep.is_dir():
                _add(d.name, dep)
            for cand in sorted(d.glob("candidate_*/deployment")):
                if cand.is_dir():
                    _add(str(cand.parent.relative_to(base_path)), cand)

        if (base_path / "candidate_1" / "deployment").is_dir():
            _add("candidate_1", base_path / "candidate_1" / "deployment")

        return deployments

    def _pick_default_iteration(self, base_path: Path, deployments: Dict[str, DeploymentInfo]) -> str:
        """Pick default deployment iteration using selection metadata when available."""
        sel = base_path / "final_submission" / "selection_metadata.json"
        if sel.exists():
            try:
                data = json.loads(sel.read_text(encoding="utf-8"))
                src = data.get("source_iteration")
                if isinstance(src, str) and src in deployments:
                    return src
            except Exception:
                pass
        return sorted(deployments.keys())[0]

    def _load_deployment_module(self, info: DeploymentInfo) -> ModuleType:
        if info.deployment_py_path is None or not info.deployment_py_path.exists():
            raise FileNotFoundError(f"deployment.py not found for iteration {info.iteration_name}")

        key = str(info.deployment_py_path.resolve())
        if key in self._module_cache:
            return self._module_cache[key]

        module_name = f"iml_deployment_{abs(hash(key))}"
        spec = importlib.util.spec_from_file_location(module_name, str(info.deployment_py_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import deployment module: {info.deployment_py_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module_cache[key] = module
        return module

    def _predict_via_deployment_module(self, data: Any, info: DeploymentInfo) -> Any:
        module = self._load_deployment_module(info)

        # Preferred API: DeploymentPredictor.load(...).predict(...)
        if hasattr(module, "DeploymentPredictor"):
            cls = getattr(module, "DeploymentPredictor")
            if hasattr(cls, "load"):
                predictor = cls.load(str(info.deployment_dir))
                if hasattr(predictor, "predict"):
                    return predictor.predict(data)

        # Fallback API: module.load(...) + module.predict(...)
        if hasattr(module, "load") and hasattr(module, "predict"):
            loaded = module.load(str(info.deployment_dir))
            return module.predict(loaded, data)

        raise AttributeError(
            f"deployment.py for iteration '{info.iteration_name}' does not expose a supported load/predict API"
        )


__all__ = ["IMLPredictor", "DeploymentInfo"]
