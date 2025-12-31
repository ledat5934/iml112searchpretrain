import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from iML.main_runner import run_automl_pipeline

def main():
    """
    Main entry point for the application when run from the terminal.
    Parses input arguments and calls the main pipeline.
    """
    parser = argparse.ArgumentParser(
        description="iML: Intelligent Machine Learning AutoML Framework",
        epilog="""
Examples:
  # Basic AutoML run
  python run.py -i ./my_dataset
  
  # Multi-iteration mode (3 different approaches)
  python run.py --checkpoint-mode multi-iteration -i ./my_dataset
  
  # Single iteration with specific algorithm type
  python run.py --single-iteration traditional -i ./my_dataset
  
  # Use different LLM provider
  python run.py -i ./my_dataset -c configs/openai_config.yaml
  
  # Checkpoint workflow - stop at guideline for manual editing
  python run.py --checkpoint-mode partial --checkpoint-action guideline -i ./dataset
  python run.py --checkpoint-mode resume --checkpoint-action preprocessing -o ./previous_run
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input data folder"
    )
    parser.add_argument(
        "-o", "--output", 
        default=None,
        help="Path to the output directory. If not provided, one will be auto-generated in the 'runs/' directory."
    )
    parser.add_argument(
        "-c", "--config", 
        default="configs/default.yaml", 
        help="Path to the configuration file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=["full", "multi-iteration", "partial", "resume"],
        default="full",
        help="Pipeline execution mode: 'full' (complete run), 'multi-iteration' (3 iterations with different algorithms), 'partial' (stop at checkpoint), 'resume' (continue from checkpoint)"
    )
    parser.add_argument(
        "--checkpoint-action",
        default="guideline",
        help="For partial mode: where to stop ('description', 'profiling', 'guideline'). For resume mode: where to start ('preprocessing', 'modeling', 'assembler')"
    )
    parser.add_argument(
        "--single-iteration",
        choices=["traditional", "custom_nn", "custom_nn_search", "pretrained"],
        default=None,
        help="Run only a single iteration approach: 'traditional' (XGBoost, LightGBM), 'custom_nn' (PyTorch no search), 'custom_nn_search' (PyTorch with architecture search), or 'pretrained' (HuggingFace models)"
    )
    parser.add_argument(
        "--ablation-variant",
        choices=["reactive", "mono", "static"],
        default=None,
        help="Run the pipeline in ablation mode to disable specific design principles: "
             "'reactive' (no Guideline agent), 'mono' (monolithic coder without modular assembly), "
             "'static' (no intermediate execution/runtime verification)."
    )
    parser.add_argument(
        "--backbone-model",
        type=str,
        choices=["gemini", "gpt-oss-20b"],
        default=None,
        help="Backbone LLM for coding agents. Options: 'gemini' (default, uses config file), "
             "'gpt-oss-20b' (4-bit quantized GPT-OSS-20B for coding agents, Gemini for others). "
             "If not specified, uses the config file default."
    )
    
    args = parser.parse_args()
    
    # Override config path if backbone-model is specified
    if args.backbone_model == "gpt-oss-20b":
        args.config = "configs/gpt-oss-20b.yaml"
        print(f"Using GPT-OSS-20B (4-bit) as coding backbone. Config: {args.config}")
    elif args.backbone_model == "gemini":
        args.config = "configs/default.yaml"
        print(f"Using Gemini as default. Config: {args.config}")
    
    # Call the main pipeline function from main_runner
    run_automl_pipeline(
        input_data_folder=args.input,
        output_folder=args.output,
        config_path=args.config,
        checkpoint_mode=args.checkpoint_mode,
        checkpoint_action=args.checkpoint_action,
        single_iteration=args.single_iteration,
        ablation_variant=args.ablation_variant,
    )

if __name__ == "__main__":
    main()