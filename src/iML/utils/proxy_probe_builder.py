import json
import textwrap
from typing import Any, Dict


def build_proxy_probe_script(
    proposal: Dict[str, Any],
    data_contract: Dict[str, Any],
    iteration_type: str | None = None,
) -> str:
    proposal_json = json.dumps(proposal or {}, ensure_ascii=False, indent=2)
    data_contract_json = json.dumps(data_contract or {}, ensure_ascii=False, indent=2)
    iteration_name = iteration_type or "default"

    template = f"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier, SGDRegressor
    from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
except Exception:
    HistGradientBoostingClassifier = None
    HistGradientBoostingRegressor = None
    RandomForestClassifier = None
    RandomForestRegressor = None
    HashingVectorizer = None
    SimpleImputer = None
    LogisticRegression = None
    Ridge = None
    SGDClassifier = None
    SGDRegressor = None
    accuracy_score = None
    f1_score = None
    log_loss = None
    mean_absolute_error = None
    mean_squared_error = None
    precision_score = None
    r2_score = None
    recall_score = None
    roc_auc_score = None
    KNeighborsClassifier = None
    KNeighborsRegressor = None
    Pipeline = None
    StandardScaler = None
    LabelEncoder = None

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None

try:
    from PIL import Image
except Exception:
    Image = None


PROPOSAL = json.loads(r'''{proposal_json}''')
DATA_CONTRACT = json.loads(r'''{data_contract_json}''')
ITERATION_TYPE = {iteration_name!r}
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CACHE_DIR = WORKSPACE_ROOT / "artifacts" / "cache"
METADATA_DIR = WORKSPACE_ROOT / "metadata"
sys.path.insert(0, str(WORKSPACE_ROOT))
from artifact_io import load_payload


def _task_kind() -> str:
    raw = str(DATA_CONTRACT.get("task_type") or "").lower()
    if "regression" in raw:
        return "regression"
    return "classification"


def _metric_preferences() -> list[str]:
    metrics = []
    modeling = DATA_CONTRACT.get("modeling_guideline") or {{}}
    for item in modeling.get("eval_metrics") or []:
        text = str(item).strip().lower()
        if text:
            metrics.append(text)
    if metrics:
        return metrics
    if _task_kind() == "regression":
        return ["rmse", "mae", "r2"]
    return ["f1", "accuracy", "roc_auc", "precision", "recall", "log_loss"]


def _normalize_score(metric_name: str, metric_value: float) -> float:
    name = str(metric_name or "").lower()
    value = float(metric_value)
    if name in {{"rmse", "mae", "mse", "log_loss"}}:
        return max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, value))))
    if name == "r2":
        return max(0.0, min(1.0, (value + 1.0) / 2.0))
    return max(0.0, min(1.0, value))


def _flatten_array(arr) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim == 0:
        out = out.reshape(1, 1)
    elif out.ndim == 1:
        out = out.reshape(-1, 1)
    elif out.ndim > 2:
        out = out.reshape(out.shape[0], -1)
    return out


def _hash_text(values) -> np.ndarray:
    if HashingVectorizer is None:
        raise RuntimeError("scikit-learn HashingVectorizer is required for text proxy probes.")
    vec = HashingVectorizer(n_features=512, alternate_sign=False, norm=None)
    return vec.transform([str(v) for v in values]).toarray().astype(np.float32)


def _image_stats(paths) -> np.ndarray:
    if Image is None:
        raise RuntimeError("Pillow is required for image-path proxy probes.")
    rows = []
    for item in paths:
        path = Path(str(item))
        if not path.exists():
            rows.append([0.0] * 8)
            continue
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                width, height = image.size
                arr = np.asarray(image.resize((32, 32)), dtype=np.float32) / 255.0
                rows.append([
                    float(width),
                    float(height),
                    float(arr.mean()),
                    float(arr.std()),
                    float(arr[..., 0].mean()),
                    float(arr[..., 1].mean()),
                    float(arr[..., 2].mean()),
                    float(path.stat().st_size),
                ])
        except Exception:
            rows.append([0.0] * 8)
    return np.asarray(rows, dtype=np.float32)


def _convert_frame(df) -> np.ndarray:
    if pd is None:
        raise RuntimeError("pandas is required to convert cached DataFrame features.")
    frame = df.copy()
    if frame.empty:
        return np.zeros((0, 1), dtype=np.float32)
    for column in frame.columns:
        if str(frame[column].dtype) in {{"object", "string", "category", "bool"}}:
            frame[column] = frame[column].astype(str).fillna("__nan__")
    try:
        frame = pd.get_dummies(frame, dummy_na=True)
    except Exception:
        frame = frame.apply(lambda col: pd.factorize(col.astype(str))[0] if str(col.dtype) in {{"object", "string", "category", "bool"}} else col)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.fillna(0.0)
    values = frame.to_numpy(dtype=np.float32, copy=False)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[1] > 2048:
        values = values[:, :2048]
    return values


def _convert_features(features) -> np.ndarray:
    if isinstance(features, np.ndarray):
        return _flatten_array(features).astype(np.float32)
    if pd is not None and isinstance(features, pd.DataFrame):
        return _convert_frame(features)
    if pd is not None and isinstance(features, pd.Series):
        return _convert_features(features.to_frame(name=features.name or "value"))
    if isinstance(features, (list, tuple)):
        if not features:
            return np.zeros((0, 1), dtype=np.float32)
        first = features[0]
        if isinstance(first, (str, os.PathLike)):
            text_values = [str(x) for x in features]
            image_like = sum(str(x).lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")) for x in text_values)
            if image_like >= max(1, int(0.8 * len(text_values))):
                return _image_stats(text_values)
            return _hash_text(text_values)
        if isinstance(first, (list, tuple, np.ndarray)):
            return _flatten_array(np.asarray(features)).astype(np.float32)
        if isinstance(first, (int, float, bool, np.number)):
            return _flatten_array(np.asarray(features)).astype(np.float32)
    return _flatten_array(np.asarray(features)).astype(np.float32)


def _convert_target(values):
    if pd is not None and isinstance(values, (pd.Series, pd.DataFrame)):
        if isinstance(values, pd.DataFrame):
            if values.shape[1] == 1:
                values = values.iloc[:, 0]
            else:
                values = values.iloc[:, 0]
        values = values.to_numpy()
    arr = np.asarray(values)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    if _task_kind() == "classification":
        if arr.dtype.kind in {{"U", "S", "O"}} or arr.dtype == bool:
            if LabelEncoder is not None:
                encoder = LabelEncoder()
                arr = encoder.fit_transform(arr.astype(str))
            else:
                uniques, encoded = np.unique(arr.astype(str), return_inverse=True)
                arr = encoded
        arr = arr.astype(np.int64)
    else:
        arr = arr.astype(np.float32)
    return arr


def _first_present(mapping: dict, keys):
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return None


def _extract_nested_split(payload, split_name: str):
    split = payload.get(split_name)
    if not isinstance(split, dict):
        return None
    x = _first_present(split, ["X", "x", "features", "data"])
    y = _first_present(split, ["y", "target", "labels"])
    return x, y


def _extract_splits(payload):
    if isinstance(payload, dict):
        direct_x_train = _first_present(payload, ["X_train", "x_train"])
        direct_y_train = _first_present(payload, ["y_train"])
        if direct_x_train is not None and direct_y_train is not None:
            x_valid = _first_present(payload, ["X_valid", "x_valid", "X_val", "x_val"])
            y_valid = _first_present(payload, ["y_valid", "y_val"])
            x_test = _first_present(payload, ["X_test", "x_test"])
            test_ids = _first_present(payload, ["test_ids", "ids_test"])
            return {{
                "X_train": direct_x_train,
                "y_train": direct_y_train,
                "X_valid": x_valid,
                "y_valid": y_valid,
                "X_test": x_test,
                "test_ids": test_ids,
            }}
        for split_keys in [("train", "valid"), ("train", "val")]:
            train_pair = _extract_nested_split(payload, split_keys[0])
            valid_pair = _extract_nested_split(payload, split_keys[1])
            if train_pair and valid_pair:
                return {{
                    "X_train": train_pair[0],
                    "y_train": train_pair[1],
                    "X_valid": valid_pair[0],
                    "y_valid": valid_pair[1],
                    "X_test": _first_present(payload.get("test") if isinstance(payload.get("test"), dict) else {{}}, ["X", "x", "features", "data"]),
                    "test_ids": _first_present(payload.get("test") if isinstance(payload.get("test"), dict) else {{}}, ["ids", "test_ids"]),
                }}
    if isinstance(payload, (list, tuple)) and len(payload) >= 4:
        return {{
            "X_train": payload[0],
            "y_train": payload[1],
            "X_valid": payload[2],
            "y_valid": payload[3],
            "X_test": payload[4] if len(payload) > 4 else None,
            "test_ids": payload[5] if len(payload) > 5 else None,
        }}
    raise RuntimeError("Unsupported cache payload structure for proxy probing. Expected standard train/valid splits.")


def _ensure_valid_split(x_train, y_train, x_valid, y_valid):
    if x_valid is not None and y_valid is not None:
        return x_train, y_train, x_valid, y_valid
    n = len(y_train)
    if n < 8:
        raise RuntimeError("Not enough rows to synthesize a validation split for proxy probing.")
    split_index = max(1, int(n * 0.8))
    return x_train[:split_index], y_train[:split_index], x_train[split_index:], y_train[split_index:]


def _subsample(x, y, limit: int):
    if len(y) <= limit:
        return x, y
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(y), size=limit, replace=False))
    return x[idx], y[idx]


def _preferred_primary_metric(all_metrics: dict) -> tuple[str, float]:
    for name in _metric_preferences():
        if name in all_metrics:
            return name, float(all_metrics[name])
    name, value = next(iter(all_metrics.items()))
    return str(name), float(value)


def _classification_metrics(y_true, pred, proba=None):
    metrics = {{
        "accuracy": float(accuracy_score(y_true, pred)) if accuracy_score is not None else 0.0,
        "f1": float(f1_score(y_true, pred, average="macro")) if f1_score is not None else 0.0,
        "precision": float(precision_score(y_true, pred, average="macro", zero_division=0)) if precision_score is not None else 0.0,
        "recall": float(recall_score(y_true, pred, average="macro", zero_division=0)) if recall_score is not None else 0.0,
    }}
    unique_count = len(np.unique(y_true))
    if proba is not None and unique_count == 2 and roc_auc_score is not None:
        try:
            if np.asarray(proba).ndim == 2 and np.asarray(proba).shape[1] >= 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, np.asarray(proba)[:, 1]))
        except Exception:
            pass
    if proba is not None and log_loss is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, proba))
        except Exception:
            pass
    return metrics


def _regression_metrics(y_true, pred):
    mse = float(mean_squared_error(y_true, pred)) if mean_squared_error is not None else 0.0
    rmse = float(np.sqrt(mse))
    return {{
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, pred)) if mean_absolute_error is not None else 0.0,
        "r2": float(r2_score(y_true, pred)) if r2_score is not None else 0.0,
        "mse": mse,
    }}


def _proposal_text() -> str:
    return " ".join([
        str(PROPOSAL.get("title") or ""),
        str(PROPOSAL.get("objective") or ""),
        str(PROPOSAL.get("rationale") or ""),
        " ".join(str(x) for x in (PROPOSAL.get("changes") or [])),
        " ".join(str(x) for x in (PROPOSAL.get("validation_focus") or [])),
    ]).lower()


def _fit_predict_sklearn(model, x_train, y_train, x_valid):
    start = time.time()
    model.fit(x_train, y_train)
    fit_seconds = time.time() - start
    pred = model.predict(x_valid)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x_valid)
        except Exception:
            proba = None
    return pred, proba, fit_seconds


def _run_linear_probe(x_train, y_train, x_valid, y_valid):
    if _task_kind() == "classification":
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("estimator", LogisticRegression(max_iter=300, solver="lbfgs")),
        ])
    else:
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("estimator", Ridge(alpha=1.0)),
        ])
    pred, proba, fit_seconds = _fit_predict_sklearn(model, x_train, y_train, x_valid)
    if _task_kind() == "classification":
        metrics = _classification_metrics(y_valid, pred, proba)
    else:
        metrics = _regression_metrics(y_valid, pred)
    name, value = _preferred_primary_metric(metrics)
    return {{
        "probe_metric_name": name,
        "probe_metric_value": value,
        "normalized_metric": _normalize_score(name, value),
        "fit_seconds": fit_seconds,
        "all_metrics": metrics,
    }}


def _run_traditional_probe(x_train, y_train, x_valid, y_valid):
    text = _proposal_text()
    if _task_kind() == "classification":
        if "forest" in text and RandomForestClassifier is not None:
            model = RandomForestClassifier(n_estimators=80, max_depth=None, random_state=42, n_jobs=-1)
        elif any(token in text for token in ["boost", "hist", "tree"]) and HistGradientBoostingClassifier is not None:
            model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, random_state=42)
        elif "knn" in text and KNeighborsClassifier is not None:
            model = KNeighborsClassifier(n_neighbors=7)
        else:
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
                ("estimator", LogisticRegression(max_iter=250, solver="lbfgs")),
            ])
    else:
        if "forest" in text and RandomForestRegressor is not None:
            model = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
        elif any(token in text for token in ["boost", "hist", "tree"]) and HistGradientBoostingRegressor is not None:
            model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, random_state=42)
        elif "knn" in text and KNeighborsRegressor is not None:
            model = KNeighborsRegressor(n_neighbors=7)
        else:
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
                ("estimator", Ridge(alpha=1.0)),
            ])
    pred, proba, fit_seconds = _fit_predict_sklearn(model, x_train, y_train, x_valid)
    metrics = _classification_metrics(y_valid, pred, proba) if _task_kind() == "classification" else _regression_metrics(y_valid, pred)
    name, value = _preferred_primary_metric(metrics)
    normalized = _normalize_score(name, value)
    throughput = float(len(y_train) / max(fit_seconds, 1e-6))
    throughput_component = min(1.0, math.log1p(throughput) / 10.0)
    probe_score = 0.85 * normalized + 0.15 * throughput_component
    return {{
        "probe_score_name": "traditional_proxy_score",
        "probe_score_value": float(probe_score),
        "primary_metric_name": name,
        "primary_metric_value": float(value),
        "all_probe_metrics": {{
            **metrics,
            "normalized_primary_metric": normalized,
            "throughput_examples_per_second": throughput,
            "fit_seconds": fit_seconds,
        }},
        "probe_details": {{
            "probe_kind": "tiny_holdout",
            "selected_model_family": type(model).__name__,
        }},
    }}


def _run_pretrained_probe(x_train, y_train, x_valid, y_valid):
    linear = _run_linear_probe(x_train, y_train, x_valid, y_valid)
    throughput = float(len(y_train) / max(linear["fit_seconds"], 1e-6))
    memory_mb = float((x_train.nbytes + x_valid.nbytes) / (1024.0 * 1024.0))
    throughput_component = min(1.0, math.log1p(throughput) / 10.0)
    memory_component = 1.0 / (1.0 + max(0.0, memory_mb) / 512.0)
    probe_score = 0.75 * linear["normalized_metric"] + 0.15 * throughput_component + 0.10 * memory_component
    return {{
        "probe_score_name": "pretrained_proxy_score",
        "probe_score_value": float(probe_score),
        "primary_metric_name": linear["probe_metric_name"],
        "primary_metric_value": float(linear["probe_metric_value"]),
        "all_probe_metrics": {{
            **linear["all_metrics"],
            "normalized_primary_metric": linear["normalized_metric"],
            "throughput_examples_per_second": throughput,
            "memory_mb": memory_mb,
            "fit_seconds": linear["fit_seconds"],
        }},
        "probe_details": {{
            "probe_kind": "linear_probe",
            "memory_component": memory_component,
            "throughput_component": throughput_component,
        }},
    }}


class _ProbeMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims, dropout: float):
        super().__init__()
        layers = []
        current = input_dim
        self.hidden_layers = nn.ModuleList()
        for hidden in hidden_dims:
            linear = nn.Linear(current, hidden)
            self.hidden_layers.append(linear)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(current, output_dim)

    def forward(self, x, return_hidden: bool = False):
        hidden = self.backbone(x)
        logits = self.head(hidden)
        if return_hidden:
            return logits, hidden
        return logits


def _run_custom_nn_probe(x_train, y_train, x_valid, y_valid):
    if torch is None or nn is None:
        fallback = _run_traditional_probe(x_train, y_train, x_valid, y_valid)
        fallback["probe_details"]["fallback_reason"] = "torch_unavailable"
        return fallback

    x_train_small, y_train_small = _subsample(x_train, y_train, 1024)
    x_valid_small, y_valid_small = _subsample(x_valid, y_valid, 512)

    train_mean = np.nanmean(x_train_small, axis=0, keepdims=True)
    train_std = np.nanstd(x_train_small, axis=0, keepdims=True)
    train_std[train_std < 1e-6] = 1.0
    x_train_small = (x_train_small - train_mean) / train_std
    x_valid_small = (x_valid_small - train_mean) / train_std

    proposal_text = _proposal_text()
    hidden_dims = [128, 64]
    if "wide" in proposal_text:
        hidden_dims = [256, 128]
    if "deep" in proposal_text:
        hidden_dims = [256, 128, 64]
    dropout = 0.0
    if "dropout" in proposal_text:
        dropout = 0.2

    input_dim = int(x_train_small.shape[1])
    if _task_kind() == "classification":
        classes = np.unique(y_train_small)
        output_dim = int(max(2, len(classes)))
    else:
        output_dim = 1

    model = _ProbeMLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x_batch = torch.tensor(x_train_small, dtype=torch.float32)
    y_batch = torch.tensor(y_train_small, dtype=torch.long if _task_kind() == "classification" else torch.float32)
    x_valid_tensor = torch.tensor(x_valid_small, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid_small, dtype=torch.long if _task_kind() == "classification" else torch.float32)

    if _task_kind() == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    model.train()
    optimizer.zero_grad()
    logits, hidden = model(x_batch, return_hidden=True)
    if _task_kind() == "classification":
        loss = criterion(logits, y_batch)
    else:
        loss = criterion(logits.reshape(-1), y_batch.reshape(-1))
    loss.backward()
    grad_norm_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad_norm_sq += float(parameter.grad.detach().pow(2).sum().cpu().item())
    grad_norm = math.sqrt(max(grad_norm_sq, 1e-12))
    grad_component = max(0.0, min(1.0, (math.log10(grad_norm + 1e-12) + 6.0) / 6.0))

    with torch.no_grad():
        active_ratio = float((hidden > 0).float().mean().cpu().item())
    activation_component = max(0.0, min(1.0, 1.0 - abs(active_ratio - 0.5) * 2.0))

    epoch_start = time.time()
    for _ in range(2):
        model.train()
        optimizer.zero_grad()
        out = model(x_batch)
        if _task_kind() == "classification":
            train_loss = criterion(out, y_batch)
        else:
            train_loss = criterion(out.reshape(-1), y_batch.reshape(-1))
        train_loss.backward()
        optimizer.step()
    fit_seconds = time.time() - epoch_start

    model.eval()
    with torch.no_grad():
        valid_logits = model(x_valid_tensor)
        if _task_kind() == "classification":
            pred = valid_logits.argmax(dim=1).cpu().numpy()
            proba = torch.softmax(valid_logits, dim=1).cpu().numpy()
            metrics = _classification_metrics(y_valid_small, pred, proba)
        else:
            pred = valid_logits.reshape(-1).cpu().numpy()
            metrics = _regression_metrics(y_valid_small, pred)

    metric_name, metric_value = _preferred_primary_metric(metrics)
    metric_component = _normalize_score(metric_name, metric_value)
    probe_score = 0.55 * metric_component + 0.25 * grad_component + 0.20 * activation_component

    return {{
        "probe_score_name": "custom_nn_proxy_score",
        "probe_score_value": float(probe_score),
        "primary_metric_name": metric_name,
        "primary_metric_value": float(metric_value),
        "all_probe_metrics": {{
            **metrics,
            "normalized_primary_metric": metric_component,
            "gradient_component": grad_component,
            "activation_component": activation_component,
            "gradient_norm": grad_norm,
            "activation_ratio": active_ratio,
            "fit_seconds": fit_seconds,
        }},
        "probe_details": {{
            "probe_kind": "zero_cost_plus_tiny_fit",
            "hidden_dims": hidden_dims,
            "dropout": dropout,
        }},
    }}


def _load_and_prepare():
    payload = load_payload(CACHE_DIR)
    splits = _extract_splits(payload)
    x_train = _convert_features(splits["X_train"])
    y_train = _convert_target(splits["y_train"])
    x_valid = _convert_features(splits["X_valid"]) if splits.get("X_valid") is not None else None
    y_valid = _convert_target(splits["y_valid"]) if splits.get("y_valid") is not None else None
    x_train, y_train, x_valid, y_valid = _ensure_valid_split(x_train, y_train, x_valid, y_valid)
    x_train, y_train = _subsample(x_train, y_train, 3000)
    x_valid, y_valid = _subsample(x_valid, y_valid, 1200)
    return x_train, y_train, x_valid, y_valid


def main():
    start = time.time()
    x_train, y_train, x_valid, y_valid = _load_and_prepare()
    if len(y_train) == 0 or len(y_valid) == 0:
        raise RuntimeError("Empty train/valid split after cache loading.")

    if ITERATION_TYPE in {{"custom_nn", "custom_nn_search"}}:
        result = _run_custom_nn_probe(x_train, y_train, x_valid, y_valid)
    elif ITERATION_TYPE == "pretrained":
        result = _run_pretrained_probe(x_train, y_train, x_valid, y_valid)
    else:
        result = _run_traditional_probe(x_train, y_train, x_valid, y_valid)

    result["proposal_id"] = PROPOSAL.get("proposal_id")
    result["iteration_type"] = ITERATION_TYPE
    result["task_kind"] = _task_kind()
    result["runtime_seconds"] = time.time() - start
    result["cache_dir"] = str(CACHE_DIR)
    result["experiment_dir"] = str(EXPERIMENT_DIR)

    output_path = EXPERIMENT_DIR / "proxy_probe_result.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"PROXY_SCORE_NAME: {{result['probe_score_name']}}")
    print(f"PROXY_SCORE_VALUE: {{float(result['probe_score_value'])}}")
    print(f"PRIMARY_METRIC_NAME: {{result['primary_metric_name']}}")
    print(f"PRIMARY_METRIC_VALUE: {{float(result['primary_metric_value'])}}")
    print(f"PROXY_RESULT_PATH: {{output_path}}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"proxy probe failed: {{exc}}", file=sys.stderr)
        raise
"""
    return textwrap.dedent(template).strip() + "\n"
