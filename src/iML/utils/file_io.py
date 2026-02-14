import os
from pathlib import Path
from typing import List

def _build_tree_conditional_limit(
    dir_path: Path,
    prefix: str,
    lines: List[str],
    csv_paths_collector: List[Path],
    root_dir: Path,
):
    """
    Recursive helper function to build the directory tree with conditional file limiting.
    - If a directory has > 50 files, it shows the first 3 and an ellipsis (...).
    - Otherwise, it shows all files.
    """
    try:
        # Keep filesystem order; do not sort
        entries = [p for p in dir_path.iterdir()]
    except OSError as e:
        lines.append(f"{prefix}└── [Error reading directory: {e}]")
        return

    # Separate directories and all files (skip description.txt)
    dirs = [e for e in entries if e.is_dir()]
    all_files = [e for e in entries if e.is_file() and e.name.lower() != "description.txt"]

    # Group files by extension (case-insensitive), keep original order
    groups_in_order = []
    groups_map = {}
    for file_entry in all_files:
        suffix_key = file_entry.suffix.lower()
        if suffix_key not in groups_map:
            groups_map[suffix_key] = []
            groups_in_order.append(suffix_key)
        groups_map[suffix_key].append(file_entry)

    # Build display list: dirs first, then file groups (max 3 per extension) + "..." per group
    entries_to_process = []
    for entry in dirs:
        entries_to_process.append((entry, True, False, entry.name))
    for suffix_key in groups_in_order:
        files_in_group = groups_map[suffix_key]
        for file_entry in files_in_group[:3]:
            entries_to_process.append((file_entry, False, False, file_entry.name))
        if len(files_in_group) > 3:
            entries_to_process.append((None, False, True, "..."))

    # Process each entry to build the tree structure
    for i, (entry, is_dir, is_ellipsis, display_name) in enumerate(entries_to_process):
        is_last_node = (i == len(entries_to_process) - 1)
        connector = "└── " if is_last_node else "├── "
        lines.append(f"{prefix}{connector}{display_name}")

        if is_dir and entry is not None:
            # If it's a directory, continue recursively with the correct prefix
            new_prefix = prefix + ("    " if is_last_node else "│   ")
            _build_tree_conditional_limit(
                entry, new_prefix, lines, csv_paths_collector, root_dir
            )

    # Go through ALL files (not just the displayed ones) to collect every CSV
    for file_entry in all_files:
        if file_entry.suffix.lower() == ".csv":
            rel_path = file_entry.relative_to(root_dir)
            if rel_path not in csv_paths_collector:
                csv_paths_collector.append(rel_path)


def get_directory_structure(
    root_dir: str,
    sample_rows: int = 5,
    include_csv_summary: bool = True,
    max_chars: int | None = 2000,
) -> str:
    """
    Generates a string representing the directory structure as a tree 
    and provides a summary of all CSV files found.

    The tree display has a conditional rule:
    - If a directory contains more than 50 files, only the first 3 are shown,
      followed by "...".
    - If it contains 50 or fewer files, all files are shown.

    Args:
        root_dir (str): Path to the root directory.
        sample_rows (int): Number of sample rows to display from each CSV file summary.
        max_chars (int | None): If set, truncate the returned string to at most this many characters.
            Use None to disable truncation.

    Returns:
        str: A string containing the directory tree and CSV summary.
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise ValueError(f"'{root_dir}' is not a valid directory.")

    # --- Part 1: Build the directory tree ---
    tree_lines: List[str] = [f"{root_path.name}/"]
    csv_paths: List[Path] = []
    
    # Call the recursive helper with the new conditional logic
    _build_tree_conditional_limit(
        dir_path=root_path,
        prefix="",
        lines=tree_lines,
        csv_paths_collector=csv_paths,
        root_dir=root_path
    )

    summary_lines: List[str] = []
    # --- Part 2: Summarize the collected CSV files ---
    if include_csv_summary and sample_rows > 0 and csv_paths:
        try:
            import pandas as pd
        except Exception as e:
            summary_lines.append("\n" + "=" * 60)
            summary_lines.append("SUMMARY OF ALL CSV FILES")
            summary_lines.append("=" * 60)
            summary_lines.append(f"\nCould not import pandas for CSV summary: {e}")
            final_output = "\n".join(tree_lines) + "\n".join(summary_lines)
            return final_output
        summary_lines.append("\n" + "="*60)
        summary_lines.append("SUMMARY OF ALL CSV FILES")
        summary_lines.append("="*60)

        csv_paths.sort()
        for rel_path in csv_paths:
            abs_path = root_path / rel_path
            try:
                df = pd.read_csv(abs_path, nrows=sample_rows, on_bad_lines='skip')
            except Exception as e:
                summary_lines.append(f"\nCould not read file '{rel_path}': {e}")
                continue
            
            summary_lines.append(f"\nStructure of file: {rel_path}")
            summary_lines.append("   Columns: " + ", ".join(df.columns.astype(str)))
            summary_lines.append("   First few rows:")
            df_string = df.to_string(index=False)
            indented_df_string = "   " + df_string.replace("\n", "\n   ")
            summary_lines.append(indented_df_string)

    # --- Part 3: Combine and return the final string ---
    final_output = "\n".join(tree_lines) + "\n".join(summary_lines)

    # Truncate to keep prompts small (while still preserving a tail context)
    if max_chars is not None and max_chars > 0 and len(final_output) > max_chars:
        # Keep head + tail to preserve both top-level layout and any ending summary lines.
        head_len = max_chars // 2
        tail_len = max_chars - head_len
        omitted = len(final_output) - max_chars
        marker = f"\n...[TRUNCATED {omitted} chars to fit max_chars={max_chars}]...\n"
        final_output = final_output[:head_len] + marker + final_output[-tail_len:]

    return final_output