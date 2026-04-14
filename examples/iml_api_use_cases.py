"""
Examples for using iML public API (`IMLPredictor`).

Run from project root:
    python examples/iml_api_use_cases.py

This file demonstrates:
1) fit() with full pipeline
2) fit() with multi-iteration pipeline
3) load() from run folder
4) load() from deployment folder
5) list_iterations() and use_iteration()
6) predict() and predict_from_file()
7) custom inference_fn override
8) basic error handling patterns
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iML import IMLPredictor, DeploymentInfo


# -----------------------------
# Helpers
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = str(PROJECT_ROOT / "configs" / "default.yaml")
DEFAULT_INPUT = str(PROJECT_ROOT / "input" / "emotion")
EXAMPLE_RUN_DIR = PROJECT_ROOT / "runs" / "example_api_run"


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _safe_print_predictions(preds: Any) -> None:
    print("Predictions type:", type(preds).__name__)
    if isinstance(preds, (list, tuple)):
        print("Predictions sample:", preds[:5])
    else:
        print("Predictions:", preds)


# -----------------------------
# Use case 1: fit() full mode
# -----------------------------
def example_fit_full() -> IMLPredictor:
    _print_header("Use case 1: fit() - full mode")

    predictor = IMLPredictor(
        path=EXAMPLE_RUN_DIR,
        config_path=DEFAULT_CONFIG,
    )

    predictor.fit(
        input_data_folder=DEFAULT_INPUT,
        checkpoint_mode="full",
    )

    print("Run dir:", predictor.run_dir)
    print("Available iterations:", predictor.list_iterations())
    print("Active iteration:", predictor.active_iteration)
    return predictor


# ------------------------------------------
# Use case 2: fit() multi-iteration mode
# ------------------------------------------
def example_fit_multi_iteration() -> IMLPredictor:
    _print_header("Use case 2: fit() - multi-iteration mode")

    run_dir = PROJECT_ROOT / "runs" / "example_api_run_multi"
    predictor = IMLPredictor(
        path=run_dir,
        config_path=DEFAULT_CONFIG,
    )

    predictor.fit(
        input_data_folder=DEFAULT_INPUT,
        checkpoint_mode="multi-iteration",
    )

    print("Run dir:", predictor.run_dir)
    print("Available iterations:", predictor.list_iterations())
    print("Active iteration:", predictor.active_iteration)
    return predictor


# ------------------------------------------
# Use case 3: load() from run folder
# ------------------------------------------
def example_load_from_run(run_dir: Path) -> IMLPredictor:
    _print_header("Use case 3: load() from run folder")

    predictor = IMLPredictor.load(run_dir)
    print("Loaded from run folder:", run_dir)
    print("Available iterations:", predictor.list_iterations())
    print("Active iteration:", predictor.active_iteration)
    return predictor


# ------------------------------------------
# Use case 4: load() from deployment folder
# ------------------------------------------
def example_load_from_deployment(deployment_dir: Path) -> IMLPredictor:
    _print_header("Use case 4: load() from deployment folder")

    predictor = IMLPredictor.load(deployment_dir)
    print("Loaded from deployment folder:", deployment_dir)
    print("Available iterations:", predictor.list_iterations())
    print("Active iteration:", predictor.active_iteration)
    return predictor


# -------------------------------------------------
# Use case 5: list_iterations() + use_iteration()
# -------------------------------------------------
def example_switch_iteration(predictor: IMLPredictor) -> None:
    _print_header("Use case 5: list_iterations() + use_iteration()")

    iterations = predictor.list_iterations()
    print("Iterations:", iterations)

    if not iterations:
        print("No deployable iterations found.")
        return

    # Switch to last iteration just to demonstrate API
    target = iterations[-1]
    predictor.use_iteration(target)
    print("Switched active iteration to:", predictor.active_iteration)


# ------------------------------------------
# Use case 6: predict() with inline data
# ------------------------------------------
def example_predict_inline(predictor: IMLPredictor) -> None:
    _print_header("Use case 6: predict() with inline data")

    sample_texts = [
        "I feel great today!",
        "This is frustrating and disappointing.",
        "It is okay, nothing special.",
    ]

    preds = predictor.predict(sample_texts)
    _safe_print_predictions(preds)


# -------------------------------------------------
# Use case 7: predict_from_file() with JSON payload
# -------------------------------------------------
def example_predict_from_file(predictor: IMLPredictor) -> None:
    _print_header("Use case 7: predict_from_file()")

    payload_file = PROJECT_ROOT / "runs" / "example_predict_payload.json"
    payload_file.write_text(
        json.dumps([
            "I am very happy.",
            "I am angry.",
            "I am calm.",
        ], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preds = predictor.predict_from_file(payload_file)
    print("Input file:", payload_file)
    _safe_print_predictions(preds)


# -------------------------------------------------
# Use case 8: custom inference_fn override
# -------------------------------------------------
def custom_inference(data: Any, info: DeploymentInfo) -> Any:
    """
    Example custom adapter:
    - Here we just print deployment location and delegate back to default module path.
    - In real projects, replace with your own strict adapter logic.
    """
    print("[custom_inference] deployment_dir:", info.deployment_dir)

    # Reuse default behavior by loading through a temporary predictor instance
    temp = IMLPredictor.load(info.deployment_dir)
    return temp.predict(data)


def example_custom_inference(run_dir: Path) -> None:
    _print_header("Use case 8: custom inference_fn")

    predictor = IMLPredictor.load(run_dir, inference_fn=custom_inference)
    preds = predictor.predict(["Custom adapter prediction call."])
    _safe_print_predictions(preds)


# ------------------------------------------
# Use case 9: explicit iteration selection
# ------------------------------------------
def example_load_with_iteration(run_dir: Path) -> None:
    _print_header("Use case 9: load(..., iteration=...)")

    p0 = IMLPredictor.load(run_dir)
    iterations = p0.list_iterations()
    if not iterations:
        print("No iterations available.")
        return

    chosen = iterations[0]
    predictor = IMLPredictor.load(run_dir, iteration=chosen)
    print("Chosen iteration:", chosen)
    print("Active iteration:", predictor.active_iteration)


# ------------------------------------------
# Use case 10: error handling pattern
# ------------------------------------------
def example_error_handling() -> None:
    _print_header("Use case 10: error handling pattern")

    try:
        _ = IMLPredictor.load(PROJECT_ROOT / "runs" / "non_existing_run")
    except Exception as e:
        print("Expected error:", repr(e))


if __name__ == "__main__":
    print("Project root:", PROJECT_ROOT)
    print("Default input:", DEFAULT_INPUT)

    # You can comment out expensive examples if needed.
    # Example minimal flow: load existing run and predict.

    # 1) Train full (can take time/cost due to LLM calls)
    # full_predictor = example_fit_full()

    # 2) Train multi-iteration (can take time/cost due to LLM calls)
    # multi_predictor = example_fit_multi_iteration()

    # For quick local demo, load a known existing run if available.
    known_run = PROJECT_ROOT / "runs" / "run_20260414_062815_a857eb51"
    if known_run.exists():
        predictor = example_load_from_run(known_run)
        example_switch_iteration(predictor)
        example_predict_inline(predictor)
        example_predict_from_file(predictor)
        example_custom_inference(known_run)
        example_load_with_iteration(known_run)

        # Deployment-folder direct load example
        if predictor.deployment is not None:
            example_load_from_deployment(predictor.deployment.deployment_dir)
    else:
        print(f"Known run not found: {known_run}")
        print("Run pipeline first or uncomment fit examples in this file.")

    example_error_handling()
