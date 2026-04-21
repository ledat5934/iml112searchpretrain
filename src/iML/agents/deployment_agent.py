import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class DeploymentAgent(BaseAgent):
    """
    Post-assembler deployment agent.

    Responsibilities:
    1) Validate that current iteration output is deployable.
    2) Write fail report in deployment folder if not deployable.
    3) Generate deployment.py exposing load()/predict()-style API without retraining.
    """

    def __init__(self, config: Dict[str, Any], manager: Any, llm_config: Dict[str, Any], max_retries: int = 2):
        super().__init__(config=config, manager=manager)
        self.llm_config = llm_config
        self.max_retries = max_retries
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="deployment_agent",
            multi_turn=llm_config.get("multi_turn", False),
        )

    def _extract_code(self, response: str) -> str:
        if "```python" in response:
            return response.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in response:
            return response.split("```", 1)[1].split("```", 1)[0].strip()
        return (response or "").strip()

    def _resolve_paths(self) -> Dict[str, Path]:
        output_dir = Path(self.manager.output_folder)
        deployment_dir = output_dir / "deployment"
        final_code = output_dir / "states" / "assemble" / "final_executable_code.py"
        if not final_code.exists():
            alt = output_dir / "states" / "final_assembled_code.py"
            if alt.exists():
                final_code = alt
        submission_path = output_dir / "submission.csv"
        return {
            "output_dir": output_dir,
            "deployment_dir": deployment_dir,
            "final_code": final_code,
            "submission_path": submission_path,
        }

    def _read_manifest(self, deployment_dir: Path) -> Dict[str, Any]:
        manifest_path = deployment_dir / "manifest.json"
        if not manifest_path.exists():
            return {"error": f"manifest.json not found at {manifest_path}"}
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"error": f"manifest.json invalid JSON: {e}"}

    def _validate_deployable(self, paths: Dict[str, Path]) -> Dict[str, Any]:
        output_dir = paths["output_dir"]
        deployment_dir = paths["deployment_dir"]
        final_code = paths["final_code"]
        submission_path = paths["submission_path"]

        problems = []
        if not final_code.exists():
            problems.append(f"final assembled code not found: {final_code}")
        if not submission_path.exists():
            problems.append(f"submission.csv not found: {submission_path}")
        if not deployment_dir.exists() or not deployment_dir.is_dir():
            problems.append(f"deployment folder not found: {deployment_dir}")

        manifest_data: Dict[str, Any] = {}
        if deployment_dir.exists() and deployment_dir.is_dir():
            manifest_data = self._read_manifest(deployment_dir)
            if "error" in manifest_data:
                problems.append(manifest_data["error"])
            else:
                referenced_files = []
                if isinstance(manifest_data, dict):
                    for value in manifest_data.values():
                        if isinstance(value, dict) and isinstance(value.get("filename"), str):
                            referenced_files.append(value["filename"])
                if not referenced_files:
                    problems.append("manifest.json has no artifact filename entries")
                else:
                    missing = [fn for fn in referenced_files if not (deployment_dir / fn).exists()]
                    if missing:
                        problems.append(f"manifest references missing artifacts: {missing[:5]}")
                    has_model = any(re.search(r"model|weights|clf|estimator", fn, flags=re.I) for fn in referenced_files)
                    has_preproc = any(re.search(r"vectorizer|tokenizer|encoder|scaler|preprocess", fn, flags=re.I) for fn in referenced_files)
                    if not has_model:
                        problems.append("manifest has no model artifact")
                    if not has_preproc:
                        problems.append("manifest has no preprocessing artifact")

        return {
            "ok": len(problems) == 0,
            "problems": problems,
            "manifest": manifest_data if isinstance(manifest_data, dict) else {},
            "output_dir": str(output_dir),
            "deployment_dir": str(deployment_dir),
            "final_code": str(final_code),
            "submission_path": str(submission_path),
        }

    def _write_fail_report(self, deployment_dir: Path, validation: Dict[str, Any]) -> None:
        deployment_dir.mkdir(parents=True, exist_ok=True)
        fail_file = deployment_dir / "deployment_fail.txt"
        content = [
            "DEPLOYMENT CHECK: FAIL",
            f"output_dir: {validation.get('output_dir')}",
            f"deployment_dir: {validation.get('deployment_dir')}",
            "problems:",
        ]
        for p in validation.get("problems", []):
            content.append(f"- {p}")
        fail_file.write_text("\n".join(content), encoding="utf-8")

    def _write_ok_report(self, deployment_dir: Path) -> None:
        deployment_dir.mkdir(parents=True, exist_ok=True)
        ok_file = deployment_dir / "deployment_ok.txt"
        ok_file.write_text("DEPLOYMENT CHECK: OK\n", encoding="utf-8")

    def _build_prompt(
        self,
        final_code: str,
        manifest: Dict[str, Any],
        deployment_dir: Path,
        iteration_type: Optional[str],
    ) -> str:
        return f"""
You are a senior Python ML deployment engineer.
Refactor the provided final_assembled_code into a reusable deployment module.

STRICT REQUIREMENTS:
1) Output ONLY one complete Python module (no explanations).
2) Module filename target: deployment.py (do not print filename, only code).
3) MUST expose AutoGluon-like API:
   - class DeploymentPredictor:
       - @classmethod load(cls, deployment_dir: str)
       - predict(self, data)
   - module-level functions:
       - load(deployment_dir: str) -> DeploymentPredictor
       - predict(predictor_or_deployment, data)
4) predict() MUST preserve preprocessing/model behavior identical to final_assembled_code.
   - Same preprocessing steps
   - Same hyperparameters and feature ordering logic
   - NO retraining under any condition
5) MUST load and use existing artifacts from deployment folder only.
6) MUST fail fast with clear errors if artifacts are missing/incompatible.
7) Keep it self-contained and executable.
8) Do NOT generate partial code, stubs, or import-only files.
9) Canonical artifact manifest contract expected in deployment/manifest.json:
     - top-level JSON object
     - each artifact entry is an object containing at least `filename`
     - recommended style example:
         {{
             "model": {{"filename": "model.joblib", "role": "model"}},
             "preprocessor": {{"filename": "preprocessor.joblib", "role": "preprocessing"}},
             "metadata": {{"filename": "metadata.json", "role": "metadata"}}
         }}
     - Avoid loose path-only styles like `model_path` / `vectorizer_path` / `model_artifact` in new outputs.

10) MANDATORY OUTPUT CONTRACT for predict():
    The task type MUST be read from deployment/metadata.json (key: "task_type", values: "classification" | "regression").
    If metadata.json does not specify task_type, infer it from the loaded model/artifacts and persist a sensible default.

    a) Classification:
       - Return a Python list with one dict per input sample, in input order:
             [
                 {{"label": <decoded_label>, "confidence": <float in [0,1]>}},
                 ...
             ]
       - `label` MUST be the decoded class label (string/int matching what training used,
         applying inverse_transform of the saved label encoder when applicable).
       - `confidence` MUST be the probability of the predicted class:
             * If the underlying model exposes `predict_proba`, take the row-wise max of the
               probability matrix:
                   probs = model.predict_proba(features)   # shape (n_samples, n_classes)
                   confidence = float(probs[i].max())      # per-sample
               This is equivalent to the probability of the predicted class because
               `model.predict()` is `classes_[argmax(predict_proba(x))]`.
             * DO NOT index probabilities by the raw predicted class value, e.g.
               `probs[i, pred_label_encoded]`. That only works when class values happen
               to be `0, 1, 2, ...`. It silently returns the wrong column (or raises
               IndexError) for any other label encoding (string labels, sparse ids,
               LabelEncoder reordering, dropped classes, etc.). Always use `.max()` or
               `argmax`-based lookup against `model.classes_` instead.
             * Else if it exposes `decision_function`, apply softmax (multiclass) or sigmoid
               (binary) to the row, then take the row-wise max as confidence.
             * Else, fall back to `confidence = 1.0` and ALSO log a clear warning to stderr.
       - Do NOT return raw numpy arrays for classification; always wrap as the list-of-dicts above.

    b) Regression:
       - Return a Python list with one dict per input sample, in input order:
             [
                 {{"value": <float>}},
                 ...
             ]
       - `value` MUST be a Python float (cast via `float(...)`), not numpy scalar.

    c) Single-sample inputs:
       - If `data` is a single sample (not a batch), still return a list of length 1 in the same shape.

    d) Errors:
       - On any inference failure, raise a clear exception. Do NOT return partial/empty results.

Iteration type: {iteration_type or 'default'}
Deployment directory: {deployment_dir}

Available manifest.json content:
```json
{json.dumps(manifest, ensure_ascii=False, indent=2)}
```

Source final_assembled_code.py:
```python
{final_code}
```
""".strip()

    def __call__(self, iteration_type: Optional[str] = None) -> Dict[str, Any]:
        self.manager.log_agent_start("DeploymentAgent: validating deployment and generating deployment.py...")

        paths = self._resolve_paths()
        deployment_dir: Path = paths["deployment_dir"]
        validation = self._validate_deployable(paths)

        try:
            self.manager.save_and_log_states(
                json.dumps(validation, ensure_ascii=False, indent=2),
                "deployment/deployment_validation.json",
            )
        except Exception:
            pass

        if not validation.get("ok"):
            self._write_fail_report(deployment_dir, validation)
            self.manager.log_agent_end("DeploymentAgent: deployment not ready (FAIL).")
            return {
                "status": "failed",
                "error": "deployment validation failed",
                "validation": validation,
            }

        final_code_path = Path(validation["final_code"])
        final_code = final_code_path.read_text(encoding="utf-8")
        manifest = validation.get("manifest", {})
        prompt = self._build_prompt(final_code, manifest, deployment_dir, iteration_type)

        self.manager.save_and_log_states(prompt, "deployment/deployment_prompt.txt")

        code = ""
        last_err = None
        for i in range(1, self.max_retries + 1):
            response = self.llm.assistant_chat(prompt)
            self.manager.save_and_log_states(response, f"deployment/attempt_{i}_raw_response.txt")
            code = self._extract_code(response)
            self.manager.save_and_log_states(code, f"deployment/attempt_{i}_deployment.py")

            if len(code.strip()) < 50:
                last_err = "deployment.py too short/empty"
                prompt += "\n\nPrevious output was empty/partial. Return full working module."
                continue

            try:
                compile(code, str(deployment_dir / "deployment.py"), "exec")
            except Exception as e:
                last_err = f"syntax error in generated deployment.py: {e}"
                prompt += f"\n\nPrevious output had syntax error: {e}. Fix and regenerate full module."
                continue

            if "def load(" not in code or "def predict(" not in code:
                last_err = "missing required load/predict API"
                prompt += "\n\nMissing required API functions load()/predict(). Regenerate."
                continue

            # basic guard against retraining in deployment.py
            forbidden = [".fit(", "fit(", "train_test_split", "model.fit", "Trainer("]
            if any(tok in code for tok in forbidden):
                last_err = "generated deployment.py appears to retrain model"
                prompt += "\n\nDo NOT retrain in deployment.py. Remove all fit/train logic."
                continue

            (deployment_dir / "deployment.py").write_text(code, encoding="utf-8")
            self._write_ok_report(deployment_dir)
            self.manager.save_and_log_states(code, "deployment/deployment.py")
            self.manager.log_agent_end("DeploymentAgent: deployment.py generated (OK).")
            return {
                "status": "success",
                "deployment_py": str(deployment_dir / "deployment.py"),
                "validation": validation,
            }

        fail_validation = {
            **validation,
            "problems": (validation.get("problems", []) + [last_err or "failed to generate deployment.py"]),
        }
        self._write_fail_report(deployment_dir, fail_validation)
        self.manager.log_agent_end("DeploymentAgent: failed to generate deployment.py.")
        return {
            "status": "failed",
            "error": last_err or "failed to generate deployment.py",
            "validation": fail_validation,
        }
