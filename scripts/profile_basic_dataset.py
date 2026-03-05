import argparse
import json
from pathlib import Path

from src.iML.utils.basic_file_profiler import (
    BasicProfilerConfig,
    build_inventory_by_extension,
    collect_sample_files_by_dir_extension,
    profile_file_basic,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Lightweight multi-format dataset profiling (dir+extension sampling).")
    ap.add_argument("--input-dir", required=True, help="Dataset root directory (the same as input_data_folder).")
    ap.add_argument("--out", default="", help="Output JSON path. If empty, prints to stdout only.")
    ap.add_argument("--max-files", type=int, default=400, help="Max total sampled files.")
    ap.add_argument("--max-file-mb", type=float, default=512.0, help="Skip files larger than this size (MB).")
    ap.add_argument("--tabular-nrows", type=int, default=50, help="Rows to sample for tabular schema (csv/tsv/parquet/xlsx/json).")
    args = ap.parse_args()

    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"--input-dir is not a directory: {root}")

    cfg = BasicProfilerConfig(
        max_files_per_dir_ext=1,
        max_total_sampled_files=int(args.max_files),
        max_file_size_mb=float(args.max_file_mb),
        tabular_nrows=int(args.tabular_nrows),
    )

    inventory = build_inventory_by_extension(root, skip_filenames={"description.txt"})
    sampled, meta = collect_sample_files_by_dir_extension(root, cfg=cfg, skip_filenames={"description.txt"})
    profiles = [profile_file_basic(p, root_dir=root, cfg=cfg) for p in sampled]

    result = {
        "input_dir": root.as_posix(),
        "basic_inventory": inventory,
        "basic_sampling_meta": meta,
        "basic_profiles": profiles,
    }

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

