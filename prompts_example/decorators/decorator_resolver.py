import logging
import re
from typing import Dict, Any, Optional, List

from .base_decorator import DecoratorChain, DomainDecorator, TaskDecorator
from .decorator_registry import DecoratorRegistry, _normalize

logger = logging.getLogger(__name__)

TASK_TYPE_TO_DOMAIN: Dict[str, str] = {
    "tabular_classification": "tabular",
    "tabular_regression": "tabular",
    "image_classification": "image",
    "text_classification": "text",
    "audio_classification": "audio",
    "video_classification": "video",
    "object_detection": "image",
    "segmentation": "image",
    "speech_recognition": "audio",
    "time_series_forecasting": "time_series",
    "pairwise_classification": "text",
    "seq2seq": "text",
    "ner": "text",
    "qa": "text",
}

TASK_TYPE_TO_TASK: Dict[str, str] = {
    "tabular_classification": "classification",
    "tabular_regression": "regression",
    "image_classification": "classification",
    "text_classification": "classification",
    "audio_classification": "classification",
    "video_classification": "classification",
    "object_detection": "object_detection",
    "segmentation": "segmentation",
    "speech_recognition": "speech_recognition",
    "time_series_forecasting": "forecasting",
    "pairwise_classification": "pairwise_classification",
    "seq2seq": "seq2seq",
    "ner": "ner",
    "qa": "seq2seq",
}

PREDICTION_TYPE_TO_TASK: Dict[str, str] = {
    "class_label": "classification",
    "probabilities": "classification",
    "boxes": "object_detection",
    "masks": "segmentation",
    "sequence": "seq2seq",
    "continuous_value": "regression",
    "time_series": "forecasting",
    "token_labels": "ner",
    "bio_tags": "ner",
}


class DecoratorResolver:

    def __init__(self, registry: Optional[DecoratorRegistry] = None):
        self.registry = registry or DecoratorRegistry()

    def resolve_from_task_schema(
        self,
        task_schema: Dict[str, Any],
        description_analysis: Dict[str, Any],
    ) -> DecoratorChain:
        domain_decorators: List[DomainDecorator] = []
        task_decorators: List[TaskDecorator] = []
        seen_domain_types = set()
        seen_task_types = set()

        modality = self._extract_modality(task_schema)
        prediction_type = self._extract_prediction_type(task_schema)
        desc_task_type = (description_analysis.get("task_type") or "").strip()

        if modality:
            modalities = self._split_modality(modality)
            for m in modalities:
                dec = self.registry.resolve_domain(m)
                if dec and type(dec) not in seen_domain_types:
                    domain_decorators.append(dec)
                    seen_domain_types.add(type(dec))

        if not domain_decorators and desc_task_type:
            domain_name = TASK_TYPE_TO_DOMAIN.get(_normalize(desc_task_type))
            if domain_name:
                dec = self.registry.resolve_domain(domain_name)
                if dec and type(dec) not in seen_domain_types:
                    domain_decorators.append(dec)
                    seen_domain_types.add(type(dec))

        if prediction_type:
            task_name = PREDICTION_TYPE_TO_TASK.get(_normalize(prediction_type), prediction_type)
            dec = self.registry.resolve_task(task_name)
            if dec and type(dec) not in seen_task_types:
                task_decorators.append(dec)
                seen_task_types.add(type(dec))

        if not task_decorators and desc_task_type:
            task_name = TASK_TYPE_TO_TASK.get(_normalize(desc_task_type), desc_task_type)
            dec = self.registry.resolve_task(task_name)
            if dec and type(dec) not in seen_task_types:
                task_decorators.append(dec)
                seen_task_types.add(type(dec))

        if not task_decorators:
            task_overview = task_schema.get("task_overview", {}) or {}
            objective = (task_overview.get("objective") or "").lower()
            if objective:
                inferred_task = self._infer_task_from_objective(objective)
                if inferred_task:
                    dec = self.registry.resolve_task(inferred_task)
                    if dec and type(dec) not in seen_task_types:
                        task_decorators.append(dec)
                        seen_task_types.add(type(dec))

        chain = DecoratorChain(
            domain_decorators=domain_decorators,
            task_decorators=task_decorators,
        )
        logger.info(f"Resolved decorator chain: {chain.summary()}")
        return chain

    def _extract_modality(self, task_schema: Dict[str, Any]) -> str:
        task_overview = task_schema.get("task_overview", {}) or {}
        return (task_overview.get("modality") or "").strip()

    def _extract_prediction_type(self, task_schema: Dict[str, Any]) -> str:
        pred_schema = task_schema.get("prediction_schema", {}) or {}
        return (pred_schema.get("type") or "").strip()

    def _split_modality(self, modality: str) -> List[str]:
        normalized = _normalize(modality)
        if normalized in ("multimodal", "multi_modal", "mixed"):
            return ["multimodal"]

        parts = re.split(r"[+/,;&|]|\band\b", modality)
        result = [p.strip() for p in parts if p.strip()]
        return result if result else [modality]

    def _infer_task_from_objective(self, objective: str) -> Optional[str]:
        keywords = {
            "classif": "classification",
            "regress": "regression",
            "detect": "object_detection",
            "segment": "segmentation",
            "forecast": "forecasting",
            "transcri": "speech_recognition",
            "translat": "seq2seq",
            "summari": "seq2seq",
            "generat": "seq2seq",
            "entity": "ner",
            "token_classif": "ner",
            "sequence_label": "ner",
        }
        for keyword, task in keywords.items():
            if keyword in objective:
                return task
        return None
