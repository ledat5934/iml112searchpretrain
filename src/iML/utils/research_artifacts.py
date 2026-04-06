import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional import in some environments
    pd = None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sanitize_name(name: str) -> str:
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch in {"_", "-", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    text = "".join(safe).strip("._")
    return text or "item"


def _dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_leaf(value: Any, cache_dir: Path, stem: str) -> dict:
    stem = _sanitize_name(stem)
    if value is None:
        return {"kind": "none"}
    if isinstance(value, (str, int, float, bool)):
        file_path = cache_dir / f"{stem}.json"
        _dump_json(file_path, {"value": value})
        return {"kind": "scalar", "path": file_path.name}
    if isinstance(value, np.ndarray):
        file_path = cache_dir / f"{stem}.npy"
        np.save(file_path, value, allow_pickle=False)
        return {"kind": "ndarray", "path": file_path.name, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)) and all(isinstance(x, (str, int, float, bool, type(None))) for x in value):
        file_path = cache_dir / f"{stem}.json"
        _dump_json(file_path, {"value": list(value)})
        return {"kind": "sequence_scalars", "path": file_path.name, "length": len(value), "sequence_type": type(value).__name__}
    if pd is not None and isinstance(value, pd.DataFrame):
        file_path = cache_dir / f"{stem}.parquet"
        value.to_parquet(file_path, index=False)
        return {"kind": "dataframe", "path": file_path.name, "shape": [int(value.shape[0]), int(value.shape[1])], "columns": [str(x) for x in value.columns]}
    if pd is not None and isinstance(value, pd.Series):
        file_path = cache_dir / f"{stem}.parquet"
        value.to_frame(name=value.name or stem).to_parquet(file_path, index=False)
        return {"kind": "series", "path": file_path.name, "length": int(len(value)), "name": value.name}

    file_path = cache_dir / f"{stem}.pkl"
    with open(file_path, "wb") as fh:
        pickle.dump(value, fh)
    return {"kind": "pickle", "path": file_path.name, "python_type": type(value).__name__}


def _save_structure(value: Any, cache_dir: Path, stem: str) -> dict:
    if isinstance(value, dict):
        children = {}
        for key, item in value.items():
            children[str(key)] = _save_structure(item, cache_dir, f"{stem}_{key}")
        return {"kind": "dict", "children": children}
    if isinstance(value, list):
        children = [_save_structure(item, cache_dir, f"{stem}_{idx}") for idx, item in enumerate(value)]
        return {"kind": "list", "children": children}
    if isinstance(value, tuple):
        children = [_save_structure(item, cache_dir, f"{stem}_{idx}") for idx, item in enumerate(value)]
        return {"kind": "tuple", "children": children}
    return _save_leaf(value, cache_dir, stem)


def save_payload(payload: Any, cache_dir: str | Path, manifest_name: str = "cache_manifest.json") -> dict:
    root = Path(cache_dir)
    _ensure_dir(root)
    manifest = _save_structure(payload, root, "payload")
    manifest_path = root / manifest_name
    _dump_json(manifest_path, manifest)
    return manifest


def _load_leaf(node: dict, cache_dir: Path) -> Any:
    kind = node.get("kind")
    if kind == "none":
        return None
    if kind == "scalar":
        raw = json.loads((cache_dir / node["path"]).read_text(encoding="utf-8"))
        return raw.get("value")
    if kind == "sequence_scalars":
        raw = json.loads((cache_dir / node["path"]).read_text(encoding="utf-8")).get("value", [])
        if node.get("sequence_type") == "tuple":
            return tuple(raw)
        return raw
    if kind == "ndarray":
        return np.load(cache_dir / node["path"], allow_pickle=False)
    if kind == "dataframe":
        if pd is None:
            raise RuntimeError("pandas is required to load cached DataFrame artifacts.")
        return pd.read_parquet(cache_dir / node["path"])
    if kind == "series":
        if pd is None:
            raise RuntimeError("pandas is required to load cached Series artifacts.")
        frame = pd.read_parquet(cache_dir / node["path"])
        column_name = frame.columns[0]
        return frame[column_name]
    if kind == "pickle":
        with open(cache_dir / node["path"], "rb") as fh:
            return pickle.load(fh)
    raise ValueError(f"Unsupported manifest leaf kind: {kind}")


def _load_structure(node: dict, cache_dir: Path) -> Any:
    kind = node.get("kind")
    if kind == "dict":
        return {key: _load_structure(child, cache_dir) for key, child in (node.get("children") or {}).items()}
    if kind == "list":
        return [_load_structure(child, cache_dir) for child in (node.get("children") or [])]
    if kind == "tuple":
        return tuple(_load_structure(child, cache_dir) for child in (node.get("children") or []))
    return _load_leaf(node, cache_dir)


def load_payload(cache_dir: str | Path, manifest_name: str = "cache_manifest.json") -> Any:
    root = Path(cache_dir)
    manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    return _load_structure(manifest, root)


ARTIFACT_IO_TEMPLATE = Path(__file__).read_text(encoding="utf-8").split("ARTIFACT_IO_TEMPLATE = ", 1)[0].rstrip()
