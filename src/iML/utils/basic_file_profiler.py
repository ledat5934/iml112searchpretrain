import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import re
import wave


@dataclass(frozen=True)
class BasicProfilerConfig:
    # Sampling rule: for each directory, for each extension, profile at most N files.
    max_files_per_dir_ext: int = 1
    # Safety caps
    max_total_sampled_files: int = 400
    max_total_dirs: int = 5000
    max_file_size_mb: float = 512.0  # skip huge files by default
    # Content sampling
    tabular_nrows: int = 50
    text_max_bytes: int = 64 * 1024
    jsonl_max_lines: int = 200
    txt_tabular_max_lines: int = 200


def _safe_relpath(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return p.as_posix()


def _iter_dirs(root: Path, max_total_dirs: int) -> Iterable[Path]:
    # Walk without sorting to keep filesystem order.
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        _ = dirnames  # not modifying traversal order
        count += 1
        if count > max_total_dirs:
            break
        yield Path(dirpath)


def collect_sample_files_by_dir_extension(
    root_dir: str | Path,
    *,
    cfg: Optional[BasicProfilerConfig] = None,
    skip_filenames: Optional[set[str]] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
    """
    Return a list of sampled file paths following the rule:
    for each directory, for each extension, take up to cfg.max_files_per_dir_ext files.
    """
    cfg = cfg or BasicProfilerConfig()
    root = Path(root_dir)
    skip_filenames = {s.lower() for s in (skip_filenames or set())}

    sampled: List[Path] = []
    notes: List[str] = []

    total_files_considered = 0
    too_big_skipped = 0

    for d in _iter_dirs(root, cfg.max_total_dirs):
        try:
            entries = [p for p in d.iterdir() if p.is_file()]
        except Exception:
            continue

        by_ext: Dict[str, List[Path]] = defaultdict(list)
        for f in entries:
            total_files_considered += 1
            if f.name.lower() in skip_filenames:
                continue
            ext = (f.suffix or "").lower()
            by_ext[ext].append(f)

        # Choose representatives
        for ext, files in by_ext.items():
            take = files[: max(1, int(cfg.max_files_per_dir_ext))]
            for f in take:
                try:
                    size_mb = f.stat().st_size / (1024 * 1024)
                except Exception:
                    size_mb = 0.0
                if size_mb > float(cfg.max_file_size_mb):
                    too_big_skipped += 1
                    continue
                sampled.append(f)
                if len(sampled) >= cfg.max_total_sampled_files:
                    notes.append(f"Reached max_total_sampled_files={cfg.max_total_sampled_files}.")
                    meta = {
                        "notes": notes,
                        "total_files_considered": total_files_considered,
                        "too_big_skipped": too_big_skipped,
                        "max_total_dirs": cfg.max_total_dirs,
                    }
                    return sampled, meta

    meta = {
        "notes": notes,
        "total_files_considered": total_files_considered,
        "too_big_skipped": too_big_skipped,
        "max_total_dirs": cfg.max_total_dirs,
    }
    return sampled, meta


def build_inventory_by_extension(root_dir: str | Path, *, skip_filenames: Optional[set[str]] = None) -> Dict[str, Any]:
    root = Path(root_dir)
    skip_filenames = {s.lower() for s in (skip_filenames or set())}

    counts_by_ext: Dict[str, int] = defaultdict(int)
    examples_by_ext: Dict[str, List[str]] = defaultdict(list)
    total_files = 0

    for d in _iter_dirs(root, max_total_dirs=5000):
        try:
            entries = [p for p in d.iterdir() if p.is_file()]
        except Exception:
            continue

        for f in entries:
            if f.name.lower() in skip_filenames:
                continue
            total_files += 1
            ext = (f.suffix or "").lower()
            counts_by_ext[ext] += 1
            if len(examples_by_ext[ext]) < 5:
                examples_by_ext[ext].append(_safe_relpath(f, root))

    # Sort extensions by count desc (stable enough)
    ext_items = sorted(counts_by_ext.items(), key=lambda kv: kv[1], reverse=True)
    counts_sorted = {k: v for k, v in ext_items}
    examples_sorted = {k: examples_by_ext[k] for k, _ in ext_items}

    return {
        "total_files": total_files,
        "counts_by_ext": counts_sorted,
        "examples_by_ext": examples_sorted,
    }


def _profile_tabular_with_pandas(path: Path, *, cfg: BasicProfilerConfig) -> Dict[str, Any]:
    try:
        import pandas as pd  # optional dependency
    except Exception as e:
        return {"error": f"pandas_not_available: {e}"}

    ext = (path.suffix or "").lower()
    read_kwargs: Dict[str, Any] = {}
    if ext in {".tsv"}:
        read_kwargs["sep"] = "\t"

    if ext in {".csv", ".tsv"}:
        df = pd.read_csv(path, nrows=cfg.tabular_nrows, on_bad_lines="skip", **read_kwargs)
    elif ext in {".parquet"}:
        # engine may be missing; catch and report
        df = pd.read_parquet(path)
        if len(df) > cfg.tabular_nrows:
            df = df.head(cfg.tabular_nrows)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=cfg.tabular_nrows)
    elif ext in {".json"}:
        # If it's a JSON lines file, it should be .jsonl; here try as normal JSON.
        df = pd.read_json(path)
        if len(df) > cfg.tabular_nrows:
            df = df.head(cfg.tabular_nrows)
    else:
        return {"error": f"unsupported_tabular_ext: {ext}"}

    dtypes = {str(c): str(t) for c, t in df.dtypes.items()}
    return {
        "nrows_sampled": int(len(df)),
        "ncols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns.tolist()],
        "dtypes": dtypes,
    }


def _profile_jsonl_keys(path: Path, *, cfg: BasicProfilerConfig) -> Dict[str, Any]:
    keys = set()
    n_ok = 0
    n_bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= cfg.jsonl_max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    keys.update(map(str, obj.keys()))
                n_ok += 1
            except Exception:
                n_bad += 1
    return {
        "jsonl_max_lines": int(cfg.jsonl_max_lines),
        "n_parsed_ok": int(n_ok),
        "n_parsed_bad": int(n_bad),
        "keys_sample": sorted(list(keys))[:200],
    }


def _profile_text_basic(path: Path, *, cfg: BasicProfilerConfig) -> Dict[str, Any]:
    # Read a small prefix; do not scan whole file.
    b = path.read_bytes()[: cfg.text_max_bytes]
    try:
        s = b.decode("utf-8", errors="replace")
    except Exception:
        s = str(b)
    lines = s.splitlines()
    return {
        "text_max_bytes": int(cfg.text_max_bytes),
        "n_lines_in_prefix": int(len(lines)),
        "prefix_preview": "\n".join(lines[:20]),
    }


def _try_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _infer_scalar_dtype(values: List[str]) -> str:
    # Very lightweight dtype inference (no pandas).
    cleaned = [v for v in (values or []) if v is not None and str(v).strip() != ""]
    if not cleaned:
        return "unknown"
    is_int = True
    is_float = True
    for v in cleaned:
        s = str(v).strip()
        if s.lower() in {"na", "nan", "null", "none"}:
            continue
        f = _try_float(s)
        if f is None:
            is_int = False
            is_float = False
            break
        # float parsed; check if int-like
        if not re.fullmatch(r"[+-]?\d+", s):
            is_int = False
    if is_int:
        return "int"
    if is_float:
        return "float"
    return "string"


def _sniff_txt_as_table(path: Path, *, cfg: BasicProfilerConfig) -> Optional[Dict[str, Any]]:
    """
    Heuristic: treat .txt as tabular if a delimiter yields a consistent column count across lines.
    We only read a small prefix.
    """
    raw = path.read_bytes()[: cfg.text_max_bytes]
    try:
        s = raw.decode("utf-8", errors="replace")
    except Exception:
        s = str(raw)

    lines_all = [ln for ln in s.splitlines() if ln.strip() != ""]
    lines = []
    for ln in lines_all:
        if ln.lstrip().startswith("#"):
            continue
        lines.append(ln)
        if len(lines) >= cfg.txt_tabular_max_lines:
            break

    if len(lines) < 5:
        return None

    candidates: List[Tuple[str, str]] = [
        ("tab", "\t"),
        ("comma", ","),
        ("semi", ";"),
        ("pipe", "|"),
        ("space", " "),  # whitespace
    ]

    best = None  # (score, name, delim, mode_cols, parsed_rows)
    for name, delim in candidates:
        parsed: List[List[str]] = []
        col_counts: List[int] = []
        for ln in lines:
            if name == "space":
                parts = [p for p in re.split(r"\s+", ln.strip()) if p != ""]
            else:
                parts = [p.strip() for p in ln.split(delim)]
            if len(parts) <= 1:
                continue
            # Cap absurdly wide rows (usually not tabular)
            if len(parts) > 300:
                continue
            parsed.append(parts)
            col_counts.append(len(parts))

        if len(col_counts) < 5:
            continue

        # mode of column count
        counts_map: Dict[int, int] = defaultdict(int)
        for c in col_counts:
            counts_map[c] += 1
        mode_cols = max(counts_map.items(), key=lambda kv: kv[1])[0]
        consistency = counts_map[mode_cols] / max(1, len(col_counts))

        # whitespace-delimited is common in natural language; require stronger evidence
        thresh = 0.9 if name == "space" else 0.85
        if mode_cols < 2 or consistency < thresh:
            continue

        # Penalize tiny mode_cols=2 with whitespace (often just "word word")
        penalty = 0.05 if (name == "space" and mode_cols <= 2) else 0.0
        score = consistency - penalty

        # keep only rows matching mode_cols for downstream dtype inference
        rows_mode = [r for r in parsed if len(r) == mode_cols]
        if best is None or score > best[0]:
            best = (score, name, delim, mode_cols, rows_mode)

    if best is None:
        return None

    _, name, delim, ncols, rows = best
    if len(rows) < 5:
        return None

    # Header heuristic: first row mostly non-numeric and later rows more numeric
    def numeric_ratio(row: List[str]) -> float:
        nums = 0
        for v in row:
            if _try_float(str(v).strip()) is not None:
                nums += 1
        return nums / max(1, len(row))

    has_header = False
    header = [f"col_{i}" for i in range(ncols)]
    if rows:
        r0 = rows[0]
        r1 = rows[1] if len(rows) > 1 else None
        r0_num = numeric_ratio(r0)
        r1_num = numeric_ratio(r1) if r1 is not None else r0_num
        # If first row is much less numeric than second row, treat it as header.
        if r0_num <= 0.2 and r1_num >= 0.4:
            has_header = True
            header = [str(x).strip() or f"col_{i}" for i, x in enumerate(r0)]
            rows = rows[1:]

    # Infer dtype per column
    cols: List[List[str]] = [[] for _ in range(ncols)]
    for r in rows[: cfg.tabular_nrows]:
        for i in range(ncols):
            cols[i].append(r[i] if i < len(r) else "")
    dtypes = {header[i]: _infer_scalar_dtype(cols[i]) for i in range(ncols)}

    return {
        "source": "txt_sniff",
        "sniff_delimiter": name,
        "ncols": int(ncols),
        "has_header": bool(has_header),
        "columns": header,
        "dtypes": dtypes,
        "nrows_sampled": int(min(len(rows), cfg.tabular_nrows)),
    }


def _profile_image_basic(path: Path) -> Dict[str, Any]:
    """
    Metadata-only image profiling (width/height/mode). Uses Pillow if available.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        return {"error": f"pillow_not_available: {e}"}

    try:
        with Image.open(path) as im:
            width, height = im.size
            mode = getattr(im, "mode", None)
            fmt = getattr(im, "format", None)
            # mode->channels mapping (approx)
            channels = None
            if mode in {"1", "L", "P"}:
                channels = 1
            elif mode in {"RGB", "YCbCr"}:
                channels = 3
            elif mode in {"RGBA", "CMYK"}:
                channels = 4
            elif isinstance(mode, str):
                channels = len(mode) if mode.isalpha() else None
            return {
                "format": fmt,
                "mode": mode,
                "width": int(width),
                "height": int(height),
                "channels": channels,
                "has_alpha": bool(mode in {"LA", "RGBA"}),
            }
    except Exception as e:
        return {"error": f"image_open_failed: {e}"}


def _profile_wav_basic(path: Path) -> Dict[str, Any]:
    """
    WAV metadata via stdlib wave (PCM WAV). Does not read full waveform.
    """
    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            duration_s = (n_frames / sample_rate) if sample_rate else None
            return {
                "n_channels": int(n_channels),
                "sample_rate": int(sample_rate),
                "sample_width_bytes": int(sampwidth),
                "n_frames": int(n_frames),
                "duration_seconds": float(duration_s) if duration_s is not None else None,
                "comptype": wf.getcomptype(),
                "compname": wf.getcompname(),
            }
    except Exception as e:
        return {"error": f"wav_open_failed: {e}"}


def profile_file_basic(
    path: str | Path,
    *,
    root_dir: Optional[str | Path] = None,
    cfg: Optional[BasicProfilerConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or BasicProfilerConfig()
    p = Path(path)
    root = Path(root_dir) if root_dir is not None else None

    ext = (p.suffix or "").lower()
    try:
        stat = p.stat()
        size_bytes = int(stat.st_size)
    except Exception:
        size_bytes = None

    base: Dict[str, Any] = {
        "path": p.as_posix(),
        "rel_path": _safe_relpath(p, root) if root is not None else p.name,
        "dir_rel": _safe_relpath(p.parent, root) if root is not None else p.parent.as_posix(),
        "extension": ext,
        "size_bytes": size_bytes,
    }

    try:
        if ext in {".png", ".jpg", ".jpeg", ".bmp"}:
            base["image_basic"] = _profile_image_basic(p)
            return base
        if ext in {".wav"}:
            base["audio_basic"] = _profile_wav_basic(p)
            return base
        if ext in {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".json"}:
            base["tabular_schema"] = _profile_tabular_with_pandas(p, cfg=cfg)
            return base
        if ext in {".jsonl"}:
            base["jsonl_schema"] = _profile_jsonl_keys(p, cfg=cfg)
            return base
        if ext in {".txt"}:
            tab = _sniff_txt_as_table(p, cfg=cfg)
            if tab is not None:
                base["tabular_schema"] = tab
            else:
                base["text_basic"] = _profile_text_basic(p, cfg=cfg)
            return base
        if ext in {".md", ".log"}:
            base["text_basic"] = _profile_text_basic(p, cfg=cfg)
            return base
    except Exception as e:
        base["error"] = f"profile_failed: {e}"
        return base

    base["note"] = "No basic profiler for this extension (kept as inventory-only)."
    return base

