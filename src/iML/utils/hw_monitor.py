import json
import os
import shutil
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil


class HardwareMonitor:
	"""Background hardware monitor for CPU/RAM/GPU usage over time."""

	def __init__(
		self,
		sample_interval_sec: float = 1.0,
		include_gpu: bool = True,
		monitor_scope: str = "host",
		target_root_pid: Optional[int] = None,
	):
		self.sample_interval_sec = max(0.2, float(sample_interval_sec))
		self.include_gpu = bool(include_gpu)
		self.monitor_scope = (monitor_scope or "host").strip().lower()
		if self.monitor_scope not in {"host", "process"}:
			self.monitor_scope = "host"
		self.target_root_pid = int(target_root_pid) if target_root_pid is not None else os.getpid()

		self._samples: List[Dict[str, Any]] = []
		self._stop_event = threading.Event()
		self._thread: Optional[threading.Thread] = None
		self._lock = threading.Lock()

		self._start_ts: Optional[float] = None
		self._iteration_name: Optional[str] = None

		self._process = psutil.Process(self.target_root_pid)

		self._nvml = None
		self._nvml_initialized = False
		self._has_nvidia_smi = shutil.which("nvidia-smi") is not None
		self._nvml_proc_util_supported: Optional[bool] = None
		self._nvml_last_seen_ts_us_by_gpu: Dict[int, int] = {}

	def start(self, iteration_name: Optional[str] = None) -> None:
		"""Start background sampling."""
		if self._thread and self._thread.is_alive():
			return

		self._iteration_name = iteration_name
		self._start_ts = time.time()
		self._stop_event.clear()
		self._samples = []

		# Prime CPU counters (first cpu_percent call is often 0/unstable)
		psutil.cpu_percent(interval=None)
		self._process.cpu_percent(interval=None)

		self._try_init_nvml()

		self._thread = threading.Thread(target=self._run, name="hardware-monitor", daemon=True)
		self._thread.start()

	def stop(self) -> None:
		"""Stop sampling thread."""
		self._stop_event.set()
		if self._thread and self._thread.is_alive():
			self._thread.join(timeout=3.0)
		self._shutdown_nvml()

	def stop_and_export(
		self,
		output_json_path: str,
		run_id: Optional[str] = None,
		iteration_name: Optional[str] = None,
		extra_metadata: Optional[Dict[str, Any]] = None,
	) -> Dict[str, Any]:
		"""Stop monitor and export samples + summary to JSON."""
		self.stop()

		with self._lock:
			samples = list(self._samples)

		resolved_iteration = iteration_name or self._iteration_name
		payload = {
			"run_id": run_id,
			"iteration": resolved_iteration,
			"monitor_scope": self.monitor_scope,
			"target_root_pid": self.target_root_pid,
			"sample_interval_sec": self.sample_interval_sec,
			"sample_count": len(samples),
			"samples": samples,
			"summary": self._build_summary(samples),
		}
		if extra_metadata:
			payload["metadata"] = extra_metadata

		output_path = Path(output_json_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
		return payload

	def _run(self) -> None:
		# Capture one sample immediately at t=0
		self._capture_sample()

		while not self._stop_event.wait(self.sample_interval_sec):
			self._capture_sample()

	def _capture_sample(self) -> None:
		now_ts = time.time()
		elapsed = (now_ts - self._start_ts) if self._start_ts else 0.0
		vm = psutil.virtual_memory()

		gpus = self._collect_gpu_metrics() if self.include_gpu else []
		gpu_util_mean = None
		total_vram_used_mb = None
		if gpus:
			gpu_util_values = [g.get("util_percent") for g in gpus if g.get("util_percent") is not None]
			vram_values = [g.get("vram_used_mb") for g in gpus if g.get("vram_used_mb") is not None]
			gpu_util_mean = sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else None
			total_vram_used_mb = sum(vram_values) if vram_values else None

		iter_gpu_util_percent = None
		iter_vram_used_mb = None
		gpu_data_source = "none"
		if self.include_gpu:
			if self.monitor_scope == "process":
				iter_metrics = self._collect_process_gpu_attribution(
					host_gpus=gpus,
					host_gpu_util_mean=gpu_util_mean,
					host_total_vram_used_mb=total_vram_used_mb,
				)
				iter_gpu_util_percent = iter_metrics.get("iter_gpu_util_percent")
				iter_vram_used_mb = iter_metrics.get("iter_vram_used_mb")
				gpu_data_source = str(iter_metrics.get("gpu_data_source") or "none")
			else:
				iter_gpu_util_percent = gpu_util_mean
				iter_vram_used_mb = total_vram_used_mb
				gpu_data_source = "host_total"

		proc_cpu = self._process.cpu_percent(interval=None)
		proc_rss_mb = round(self._get_process_tree_rss_bytes() / (1024 * 1024), 2)

		sample = {
			"ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
			"elapsed_sec": round(elapsed, 3),
			"cpu_percent": psutil.cpu_percent(interval=None),
			"ram_used_mb": round(vm.used / (1024 * 1024), 2),
			"ram_percent": vm.percent,
			"proc_cpu_percent": proc_cpu,
			"proc_rss_mb": proc_rss_mb,
			"gpus": gpus,
			"gpu_util_mean_percent": round(gpu_util_mean, 3) if gpu_util_mean is not None else None,
			"vram_used_total_mb": round(total_vram_used_mb, 3) if total_vram_used_mb is not None else None,
			"iter_cpu_percent": proc_cpu if self.monitor_scope == "process" else None,
			"iter_ram_used_mb": proc_rss_mb if self.monitor_scope == "process" else None,
			"iter_gpu_util_percent": round(iter_gpu_util_percent, 3) if iter_gpu_util_percent is not None else None,
			"iter_vram_used_mb": round(iter_vram_used_mb, 3) if iter_vram_used_mb is not None else None,
			"gpu_data_source": gpu_data_source,
		}

		with self._lock:
			self._samples.append(sample)

	def _get_process_tree_rss_bytes(self) -> int:
		total = 0
		try:
			total += self._process.memory_info().rss
			for child in self._process.children(recursive=True):
				try:
					total += child.memory_info().rss
				except Exception:
					continue
		except Exception:
			pass
		return total

	def _try_init_nvml(self) -> None:
		if not self.include_gpu:
			return
		try:
			import pynvml  # type: ignore

			pynvml.nvmlInit()
			self._nvml = pynvml
			self._nvml_initialized = True
		except Exception:
			self._nvml = None
			self._nvml_initialized = False

	def _shutdown_nvml(self) -> None:
		if self._nvml_initialized and self._nvml is not None:
			try:
				self._nvml.nvmlShutdown()
			except Exception:
				pass
		self._nvml = None
		self._nvml_initialized = False

	def _collect_gpu_metrics(self) -> List[Dict[str, Any]]:
		if self._nvml_initialized:
			metrics = self._collect_gpu_metrics_nvml()
			if metrics:
				return metrics

		if self._has_nvidia_smi:
			return self._collect_gpu_metrics_nvidia_smi()

		return []

	def _collect_gpu_metrics_nvml(self) -> List[Dict[str, Any]]:
		pynvml = self._nvml
		if pynvml is None:
			return []

		out: List[Dict[str, Any]] = []
		try:
			device_count = pynvml.nvmlDeviceGetCount()
			for idx in range(device_count):
				handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
				util = pynvml.nvmlDeviceGetUtilizationRates(handle)
				mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

				temperature = None
				try:
					temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
				except Exception:
					temperature = None

				power_w = None
				try:
					power_w = round(float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0, 3)
				except Exception:
					power_w = None

				out.append(
					{
						"index": idx,
						"util_percent": float(util.gpu),
						"vram_used_mb": round(mem.used / (1024 * 1024), 2),
						"vram_total_mb": round(mem.total / (1024 * 1024), 2),
						"temperature_c": temperature,
						"power_w": power_w,
					}
				)
		except Exception:
			return []
		return out

	def _collect_gpu_metrics_nvidia_smi(self) -> List[Dict[str, Any]]:
		try:
			cmd = [
				"nvidia-smi",
				"--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
				"--format=csv,noheader,nounits",
			]
			proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
			if proc.returncode != 0:
				return []

			rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
			parsed: List[Dict[str, Any]] = []
			for row in rows:
				parts = [p.strip() for p in row.split(",")]
				if len(parts) < 6:
					continue
				parsed.append(
					{
						"index": int(float(parts[0])),
						"util_percent": float(parts[1]),
						"vram_used_mb": float(parts[2]),
						"vram_total_mb": float(parts[3]),
						"temperature_c": float(parts[4]) if parts[4] else None,
						"power_w": float(parts[5]) if parts[5] else None,
					}
				)
			return parsed
		except Exception:
			return []

	def _get_pid_tree_set(self) -> set[int]:
		out: set[int] = set()
		try:
			out.add(int(self._process.pid))
			for child in self._process.children(recursive=True):
				try:
					out.add(int(child.pid))
				except Exception:
					continue
		except Exception:
			pass
		return out

	def _collect_process_gpu_attribution(
		self,
		host_gpus: List[Dict[str, Any]],
		host_gpu_util_mean: Optional[float],
		host_total_vram_used_mb: Optional[float],
	) -> Dict[str, Any]:
		pid_set = self._get_pid_tree_set()
		if not pid_set:
			return {
				"iter_gpu_util_percent": 0.0,
				"iter_vram_used_mb": 0.0,
				"gpu_data_source": "none",
			}

		vram_nvml = self._collect_pid_vram_from_nvml(pid_set)
		vram_any = vram_nvml
		if vram_any is None:
			vram_any = self._collect_pid_vram_from_nvidia_smi_query(pid_set)
		util_nvml = self._collect_pid_util_from_nvml(pid_set)

		if vram_any is not None and util_nvml is not None:
			return {
				"iter_gpu_util_percent": float(util_nvml),
				"iter_vram_used_mb": float(vram_any),
				"gpu_data_source": "nvml_per_pid",
			}

		util_pmon = self._collect_pid_util_from_pmon(pid_set)
		if vram_any is not None and util_pmon is not None:
			return {
				"iter_gpu_util_percent": float(util_pmon),
				"iter_vram_used_mb": float(vram_any),
				"gpu_data_source": "pmon_per_pid",
			}

		if vram_any is not None and host_gpu_util_mean is not None and host_total_vram_used_mb and host_total_vram_used_mb > 0:
			est = float(host_gpu_util_mean) * (float(vram_any) / float(host_total_vram_used_mb))
			return {
				"iter_gpu_util_percent": max(0.0, est),
				"iter_vram_used_mb": float(vram_any),
				"gpu_data_source": "estimated_by_vram_share",
			}

		return {
			"iter_gpu_util_percent": 0.0,
			"iter_vram_used_mb": float(vram_any) if vram_any is not None else 0.0,
			"gpu_data_source": "none",
		}

	def _collect_pid_vram_from_nvml(self, pid_set: set[int]) -> Optional[float]:
		if not self._nvml_initialized or self._nvml is None:
			return None
		pynvml = self._nvml
		total_mb = 0.0
		found_any = False
		try:
			device_count = pynvml.nvmlDeviceGetCount()
			for idx in range(device_count):
				handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
				proc_lists: List[Any] = []
				for fn_name in (
					"nvmlDeviceGetComputeRunningProcesses_v2",
					"nvmlDeviceGetComputeRunningProcesses",
					"nvmlDeviceGetGraphicsRunningProcesses_v2",
					"nvmlDeviceGetGraphicsRunningProcesses",
				):
					fn = getattr(pynvml, fn_name, None)
					if fn is None:
						continue
					try:
						proc_lists.extend(fn(handle) or [])
					except Exception:
						continue

				for p in proc_lists:
					try:
						pid = int(getattr(p, "pid", -1))
					except Exception:
						pid = -1
					if pid not in pid_set:
						continue
					used = getattr(p, "usedGpuMemory", None)
					if used is None:
						continue
					try:
						used_int = int(used)
					except Exception:
						continue
					if used_int < 0:
						continue
					total_mb += float(used_int) / (1024 * 1024)
					found_any = True
		except Exception:
			return None
		return total_mb if found_any else 0.0

	def _collect_pid_util_from_nvml(self, pid_set: set[int]) -> Optional[float]:
		if not self._nvml_initialized or self._nvml is None:
			return None
		pynvml = self._nvml
		total_util = 0.0
		found_any = False
		try:
			device_count = pynvml.nvmlDeviceGetCount()
			for idx in range(device_count):
				handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
				fn = getattr(pynvml, "nvmlDeviceGetProcessUtilization", None)
				if fn is None:
					self._nvml_proc_util_supported = False
					return None
				last_ts = int(self._nvml_last_seen_ts_us_by_gpu.get(idx, 0))
				try:
					samples = fn(handle, last_ts) or []
				except Exception:
					self._nvml_proc_util_supported = False
					return None

				max_seen_ts = last_ts
				per_gpu_util = 0.0
				for s in samples:
					ts = int(getattr(s, "timeStamp", 0) or 0)
					if ts > max_seen_ts:
						max_seen_ts = ts
					try:
						pid = int(getattr(s, "pid", -1))
					except Exception:
						pid = -1
					if pid not in pid_set:
						continue
					sm_util = getattr(s, "smUtil", None)
					if sm_util is None:
						continue
					try:
						per_gpu_util += float(sm_util)
						found_any = True
					except Exception:
						continue

				self._nvml_last_seen_ts_us_by_gpu[idx] = max_seen_ts
				total_util += max(0.0, per_gpu_util)
			self._nvml_proc_util_supported = True
		except Exception:
			return None
		if not found_any:
			return 0.0
		return total_util

	def _collect_pid_util_from_pmon(self, pid_set: set[int]) -> Optional[float]:
		if not self._has_nvidia_smi:
			return None
		try:
			cmd = ["nvidia-smi", "pmon", "-c", "1", "-s", "um"]
			proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
			if proc.returncode != 0:
				return None

			total_sm = 0.0
			found = False
			for raw in proc.stdout.splitlines():
				line = raw.strip()
				if not line or line.startswith("#"):
					continue
				parts = line.split()
				if len(parts) < 5:
					continue
				try:
					pid = int(parts[1])
				except Exception:
					continue
				if pid <= 0 or pid not in pid_set:
					continue
				sm_raw = parts[3]
				if sm_raw == "-":
					continue
				try:
					total_sm += float(sm_raw)
					found = True
				except Exception:
					continue
			if not found:
				return 0.0
			return total_sm
		except Exception:
			return None

	def _collect_pid_vram_from_nvidia_smi_query(self, pid_set: set[int]) -> Optional[float]:
		"""Fallback VRAM attribution via nvidia-smi compute-apps query.

		Returns 0.0 when command is available but none of the target PIDs are using GPU.
		Returns None when command is unavailable or fails.
		"""
		if not self._has_nvidia_smi:
			return None
		try:
			cmd = [
				"nvidia-smi",
				"--query-compute-apps=pid,used_memory",
				"--format=csv,noheader,nounits",
			]
			proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
			if proc.returncode != 0:
				return None

			total_mb = 0.0
			# Empty output is valid: no active compute processes.
			for raw in proc.stdout.splitlines():
				line = raw.strip()
				if not line:
					continue
				parts = [p.strip() for p in line.split(",")]
				if len(parts) < 2:
					continue
				try:
					pid = int(parts[0])
				except Exception:
					continue
				if pid not in pid_set:
					continue
				try:
					total_mb += float(parts[1])
				except Exception:
					continue
			return total_mb
		except Exception:
			return None

	def _build_summary(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
		if not samples:
			return {
				"duration_sec": 0.0,
				"cpu_peak_percent": None,
				"ram_peak_mb": None,
				"proc_cpu_peak_percent": None,
				"proc_rss_peak_mb": None,
				"gpu_util_peak_percent": None,
				"vram_peak_mb": None,
			}

		cpu_vals = [s.get("cpu_percent") for s in samples if s.get("cpu_percent") is not None]
		ram_vals = [s.get("ram_used_mb") for s in samples if s.get("ram_used_mb") is not None]
		proc_cpu_vals = [s.get("proc_cpu_percent") for s in samples if s.get("proc_cpu_percent") is not None]
		proc_rss_vals = [s.get("proc_rss_mb") for s in samples if s.get("proc_rss_mb") is not None]
		gpu_util_vals = [s.get("gpu_util_mean_percent") for s in samples if s.get("gpu_util_mean_percent") is not None]
		vram_vals = [s.get("vram_used_total_mb") for s in samples if s.get("vram_used_total_mb") is not None]
		iter_gpu_vals = [s.get("iter_gpu_util_percent") for s in samples if s.get("iter_gpu_util_percent") is not None]
		iter_vram_vals = [s.get("iter_vram_used_mb") for s in samples if s.get("iter_vram_used_mb") is not None]
		source_counts = Counter(
			[str(s.get("gpu_data_source")) for s in samples if s.get("gpu_data_source")]
		)

		duration = float(samples[-1].get("elapsed_sec") or 0.0)

		return {
			"duration_sec": round(duration, 3),
			"cpu_peak_percent": max(cpu_vals) if cpu_vals else None,
			"ram_peak_mb": max(ram_vals) if ram_vals else None,
			"proc_cpu_peak_percent": max(proc_cpu_vals) if proc_cpu_vals else None,
			"proc_rss_peak_mb": max(proc_rss_vals) if proc_rss_vals else None,
			"gpu_util_peak_percent": max(gpu_util_vals) if gpu_util_vals else None,
			"vram_peak_mb": max(vram_vals) if vram_vals else None,
			"iter_gpu_util_peak_percent": max(iter_gpu_vals) if iter_gpu_vals else None,
			"iter_vram_peak_mb": max(iter_vram_vals) if iter_vram_vals else None,
			"gpu_data_source_counts": dict(source_counts),
		}
