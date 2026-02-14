import argparse
import sys
from pathlib import Path

from src.iML.utils.file_io import get_directory_structure


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print dataset directory structure with grouped file listing."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to dataset root (the folder containing description.txt).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="Number of rows to sample from each CSV for summary.",
    )
    parser.add_argument(
        "--no-csv-summary",
        action="store_true",
        help="Skip CSV summaries (no pandas required).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum characters to print (use 0 to disable truncation).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Path not found: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dataset_path}")

    output = get_directory_structure(
        str(dataset_path),
        sample_rows=args.sample_rows,
        include_csv_summary=not args.no_csv_summary,
        max_chars=None if args.max_chars == 0 else args.max_chars,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
