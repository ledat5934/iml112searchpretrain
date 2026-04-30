#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Plot aligned hardware usage from run-level and per-iteration JSON files.

Usage:
    python scripts/plot_hw_usage.py --run-dir runs/run_20260421_123456_abcd1234
"""

import argparse
import importlib
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

plt = importlib.import_module("matplotlib.pyplot")


ITER_ORDER = ["traditional", "custom_nn_search", "pretrained"]


def _parse_ts_to_epoch(ts: str) -> Optional[float]:
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None


def _load_iteration_usage(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for iteration_dir in sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("iteration_")]):
        usage_file = iteration_dir / "hardware_usage.json"
        if not usage_file.exists():
            continue
        try:
            payload = json.loads(usage_file.read_text(encoding="utf-8"))
            iteration_name = str(payload.get("iteration") or iteration_dir.name)
            results[iteration_name] = payload
        except Exception as e:
            print(f"[WARN] Could not read {usage_file}: {e}")
    return results


def _load_overall_usage(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "hardware_usage_total.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Could not read {p}: {e}")
        return None


def _sample_epoch(s: Dict[str, Any]) -> Optional[float]:
    ts = s.get("ts")
    if isinstance(ts, str):
        return _parse_ts_to_epoch(ts)
    return None


def _collect_all_epochs(overall: Optional[Dict[str, Any]], iters: Dict[str, Dict[str, Any]]) -> List[float]:
    out: List[float] = []
    if overall:
        for s in (overall.get("samples", []) or []):
            ep = _sample_epoch(s)
            if ep is not None:
                out.append(ep)
    for payload in iters.values():
        for s in (payload.get("samples", []) or []):
            ep = _sample_epoch(s)
            if ep is not None:
                out.append(ep)
    return out


def _extract_points(samples: List[Dict[str, Any]], key: str, t0: float) -> List[Tuple[int, float]]:
    pts: List[Tuple[int, float]] = []
    for s in samples:
        y = s.get(key)
        ep = _sample_epoch(s)
        if y is None or ep is None:
            continue
        sec = int(round(ep - t0))
        if sec < 0:
            continue
        pts.append((sec, float(y)))
    pts.sort(key=lambda x: x[0])
    return pts


def _extract_points_by_fn(
    samples: List[Dict[str, Any]],
    t0: float,
    value_fn,
) -> List[Tuple[int, float]]:
    pts: List[Tuple[int, float]] = []
    for s in samples:
        y = value_fn(s)
        ep = _sample_epoch(s)
        if y is None or ep is None:
            continue
        sec = int(round(ep - t0))
        if sec < 0:
            continue
        pts.append((sec, float(y)))
    pts.sort(key=lambda x: x[0])
    return pts


def _dense_series(points: List[Tuple[int, float]], t_end: int, zero_outside: bool) -> Tuple[List[float], List[float]]:
    xs = [float(s) for s in range(0, t_end + 1)]
    if not points:
        return xs, [0.0 for _ in xs]

    first_sec = points[0][0]
    last_sec = points[-1][0]
    i = 0
    last_val: Optional[float] = None
    ys: List[float] = []
    for sec in range(0, t_end + 1):
        while i < len(points) and points[i][0] <= sec:
            last_val = points[i][1]
            i += 1

        if zero_outside and (sec < first_sec or sec > last_sec):
            ys.append(0.0)
        else:
            if last_val is None:
                ys.append(points[0][1])
            else:
                ys.append(last_val)
    return xs, ys


def _gpu_source_label(payload: Dict[str, Any]) -> str:
    summary = payload.get("summary", {}) or {}
    counts = summary.get("gpu_data_source_counts") or {}
    if not counts:
        counts = {}
        for s in (payload.get("samples", []) or []):
            src = s.get("gpu_data_source")
            if not src:
                continue
            counts[src] = int(counts.get(src, 0)) + 1
    if not counts:
        return "unknown"
    return str(max(counts.items(), key=lambda kv: kv[1])[0])


def _estimate_total_ram_mb(overall: Optional[Dict[str, Any]], iters: Dict[str, Dict[str, Any]]) -> Optional[float]:
    candidates = []
    payloads: List[Dict[str, Any]] = ([] if overall is None else [overall]) + list(iters.values())
    for payload in payloads:
        for s in (payload.get("samples", []) or []):
            used = s.get("ram_used_mb")
            pct = s.get("ram_percent")
            if used is None or pct is None:
                continue
            try:
                used_f = float(used)
                pct_f = float(pct)
                if pct_f > 0:
                    candidates.append(used_f * 100.0 / pct_f)
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort()
    return candidates[len(candidates) // 2]


def _estimate_total_vram_mb(overall: Optional[Dict[str, Any]], iters: Dict[str, Dict[str, Any]]) -> Optional[float]:
    payloads: List[Dict[str, Any]] = ([] if overall is None else [overall]) + list(iters.values())
    for payload in payloads:
        for s in (payload.get("samples", []) or []):
            gpus = s.get("gpus") or []
            if not isinstance(gpus, list) or not gpus:
                continue
            total = 0.0
            ok = False
            for g in gpus:
                if not isinstance(g, dict):
                    continue
                v = g.get("vram_total_mb")
                if v is None:
                    continue
                try:
                    total += float(v)
                    ok = True
                except Exception:
                    continue
            if ok and total > 0:
                return total
    return None


def _plot(run_dir: Path, output_file: Path) -> None:
    iteration_data = _load_iteration_usage(run_dir)
    overall = _load_overall_usage(run_dir)
    if not iteration_data and overall is None:
        raise RuntimeError(f"No hardware_usage.json found under {run_dir}")

    all_epochs = _collect_all_epochs(overall, iteration_data)
    if not all_epochs:
        raise RuntimeError(f"No timestamp samples found under {run_dir}")

    t0 = min(all_epochs)
    t_end = int(round(max(all_epochs) - t0))
    total_ram_mb = _estimate_total_ram_mb(overall, iteration_data)
    total_vram_mb = _estimate_total_vram_mb(overall, iteration_data)

    def dense_overall(key: str) -> Tuple[List[float], List[float]]:
        if overall is None:
            return [float(s) for s in range(0, t_end + 1)], [0.0 for _ in range(0, t_end + 1)]
        pts = _extract_points(overall.get("samples", []) or [], key, t0)
        return _dense_series(pts, t_end=t_end, zero_outside=False)

    def dense_iter(iter_name: str, key_candidates: List[str]) -> Tuple[List[float], List[float]]:
        payload = iteration_data.get(iter_name)
        if payload is None:
            return [float(s) for s in range(0, t_end + 1)], [0.0 for _ in range(0, t_end + 1)]
        samples = payload.get("samples", []) or []
        for k in key_candidates:
            pts = _extract_points(samples, k, t0)
            if pts:
                return _dense_series(pts, t_end=t_end, zero_outside=True)
        return [float(s) for s in range(0, t_end + 1)], [0.0 for _ in range(0, t_end + 1)]

    def dense_overall_ram_percent() -> Tuple[List[float], List[float]]:
        if overall is None:
            return [float(s) for s in range(0, t_end + 1)], [0.0 for _ in range(0, t_end + 1)]
        pts = _extract_points(overall.get("samples", []) or [], "ram_percent", t0)
        return _dense_series(pts, t_end=t_end, zero_outside=False)

    def dense_iter_ram_percent(iter_name: str) -> Tuple[List[float], List[float]]:
        payload = iteration_data.get(iter_name)
        if payload is None or not total_ram_mb or total_ram_mb <= 0:
            return [float(s) for s in range(0, t_end + 1)], [0.0 for _ in range(0, t_end + 1)]
        samples = payload.get("samples", []) or []

        def _ram_fn(s: Dict[str, Any]) -> Optional[float]:
            v = s.get("iter_ram_used_mb")
            if v is None:
                v = s.get("proc_rss_mb")
            if v is None:
                return None
            try:
                return max(0.0, min(100.0, float(v) * 100.0 / float(total_ram_mb)))
            except Exception:
                return None

        pts = _extract_points_by_fn(samples, t0, _ram_fn)
        return _dense_series(pts, t_end=t_end, zero_outside=True)

    def dense_overall_vram_percent() -> Tuple[List[float], List[float]]:
        if overall is None or not total_vram_mb or total_vram_mb <= 0:
            return [float(s) for s in range(0, t_end + 1)], [0.0 for _ in range(0, t_end + 1)]
        samples = overall.get("samples", []) or []

        def _vram_fn(s: Dict[str, Any]) -> Optional[float]:
            v = s.get("vram_used_total_mb")
            if v is None:
                return None
            try:
                return max(0.0, min(100.0, float(v) * 100.0 / float(total_vram_mb)))
            except Exception:
                return None

        pts = _extract_points_by_fn(samples, t0, _vram_fn)
        return _dense_series(pts, t_end=t_end, zero_outside=False)

    def dense_iter_vram_percent(iter_name: str) -> Tuple[List[float], List[float]]:
        payload = iteration_data.get(iter_name)
        if payload is None or not total_vram_mb or total_vram_mb <= 0:
            return [float(s) for s in range(0, t_end + 1)], [0.0 for _ in range(0, t_end + 1)]
        samples = payload.get("samples", []) or []

        def _iter_vram_fn(s: Dict[str, Any]) -> Optional[float]:
            v = s.get("iter_vram_used_mb")
            if v is None:
                v = s.get("vram_used_total_mb")
            if v is None:
                return None
            try:
                return max(0.0, min(100.0, float(v) * 100.0 / float(total_vram_mb)))
            except Exception:
                return None

        pts = _extract_points_by_fn(samples, t0, _iter_vram_fn)
        return _dense_series(pts, t_end=t_end, zero_outside=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False)
    ax_cpu = axes[0][0]
    ax_ram = axes[0][1]
    ax_gpu = axes[1][0]
    ax_vram = axes[1][1]

    # Overall lines (host-level)
    x_cpu_o, y_cpu_o = dense_overall("cpu_percent")
    x_ram_o, y_ram_o = dense_overall_ram_percent()
    x_gpu_o, y_gpu_o = dense_overall("gpu_util_mean_percent")
    x_vram_o, y_vram_o = dense_overall_vram_percent()
    ax_cpu.plot(x_cpu_o, y_cpu_o, label="overall", linewidth=2.2)
    ax_ram.plot(x_ram_o, y_ram_o, label="overall", linewidth=2.2)
    ax_gpu.plot(x_gpu_o, y_gpu_o, label="overall", linewidth=2.2)
    ax_vram.plot(x_vram_o, y_vram_o, label="overall", linewidth=2.2)

    source_labels: List[str] = []
    for iter_name in ITER_ORDER:
        x_cpu, y_cpu = dense_iter(iter_name, ["iter_cpu_percent", "proc_cpu_percent"])
        x_ram, y_ram = dense_iter_ram_percent(iter_name)
        x_gpu, y_gpu = dense_iter(iter_name, ["iter_gpu_util_percent", "gpu_util_mean_percent"])
        x_vram, y_vram = dense_iter_vram_percent(iter_name)

        payload = iteration_data.get(iter_name)
        src = _gpu_source_label(payload) if payload is not None else "missing"
        source_labels.append(f"{iter_name}: {src}")

        ax_cpu.plot(x_cpu, y_cpu, label=iter_name)
        ax_ram.plot(x_ram, y_ram, label=iter_name)
        ax_gpu.plot(x_gpu, y_gpu, label=f"{iter_name} ({src})")
        ax_vram.plot(x_vram, y_vram, label=f"{iter_name} ({src})")

    ax_cpu.set_title("CPU usage (%)")
    ax_cpu.set_xlabel("Time since pipeline start (s)")
    ax_cpu.set_ylabel("CPU %")
    ax_cpu.grid(True, alpha=0.3)
    ax_cpu.legend()

    ax_ram.set_title("RAM usage (%)")
    ax_ram.set_xlabel("Time since pipeline start (s)")
    ax_ram.set_ylabel("RAM %")
    ax_ram.grid(True, alpha=0.3)
    ax_ram.legend()

    ax_gpu.set_title("GPU usage mean (%)")
    ax_gpu.set_xlabel("Time since pipeline start (s)")
    ax_gpu.set_ylabel("GPU util %")
    ax_gpu.grid(True, alpha=0.3)
    ax_gpu.legend()

    ax_vram.set_title("VRAM usage (%)")
    ax_vram.set_xlabel("Time since pipeline start (s)")
    ax_vram.set_ylabel("VRAM %")
    ax_vram.grid(True, alpha=0.3)
    ax_vram.legend()

    for ax in (ax_cpu, ax_ram, ax_gpu, ax_vram):
        ax.set_ylim(0, 100)

    fig.suptitle(f"Hardware usage by iteration (aligned to pipeline start)\n{run_dir.name}")
    if source_labels:
        fig.text(
            0.5,
            0.01,
            "GPU/VRAM per-iteration source: " + " | ".join(source_labels),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=160)
    print(f"[OK] Saved plot: {output_file}")
    if source_labels:
        print("[INFO] GPU/VRAM per-iteration source:")
        for line in source_labels:
            print(f"  - {line}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot hardware usage from per-iteration JSON logs.")
    parser.add_argument("--run-dir", required=True, help="Path to run folder (e.g., runs/run_20260421_...")
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG file path (default: <run-dir>/hardware_usage_overview.png)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    output = Path(args.output).resolve() if args.output else (run_dir / "hardware_usage_overview.png")
    _plot(run_dir, output)
