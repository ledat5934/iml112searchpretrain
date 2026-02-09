import difflib
import logging
import re
from typing import Optional, Dict, Type

from .base_decorator import DomainDecorator, TaskDecorator
from .domain_decorators import (
    TabularDomainDecorator,
    CVDomainDecorator,
    NLPDomainDecorator,
    AudioDomainDecorator,
    VideoDomainDecorator,
    TimeSeriesDomainDecorator,
    MultimodalDomainDecorator,
)
from .task_decorators import (
    ClassificationTaskDecorator,
    RegressionTaskDecorator,
    ObjectDetectionTaskDecorator,
    SegmentationTaskDecorator,
    SpeechRecognitionTaskDecorator,
    TimeSeriesForecastingTaskDecorator,
    PairwiseClassificationTaskDecorator,
    Seq2SeqTaskDecorator,
    NERTaskDecorator,
)

logger = logging.getLogger(__name__)


def _normalize(name: str) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class DecoratorRegistry:

    DOMAIN_MAP: Dict[str, Type[DomainDecorator]] = {
        "tabular": TabularDomainDecorator,
        "csv": TabularDomainDecorator,
        "structured": TabularDomainDecorator,
        "structured_data": TabularDomainDecorator,
        "table": TabularDomainDecorator,
        "dataframe": TabularDomainDecorator,
        "image": CVDomainDecorator,
        "cv": CVDomainDecorator,
        "computer_vision": CVDomainDecorator,
        "vision": CVDomainDecorator,
        "image_data": CVDomainDecorator,
        "images": CVDomainDecorator,
        "text": NLPDomainDecorator,
        "nlp": NLPDomainDecorator,
        "natural_language": NLPDomainDecorator,
        "natural_language_processing": NLPDomainDecorator,
        "language": NLPDomainDecorator,
        "textual": NLPDomainDecorator,
        "audio": AudioDomainDecorator,
        "speech": AudioDomainDecorator,
        "sound": AudioDomainDecorator,
        "audio_data": AudioDomainDecorator,
        "video": VideoDomainDecorator,
        "video_data": VideoDomainDecorator,
        "time_series": TimeSeriesDomainDecorator,
        "timeseries": TimeSeriesDomainDecorator,
        "temporal": TimeSeriesDomainDecorator,
        "time_series_data": TimeSeriesDomainDecorator,
        "sequential": TimeSeriesDomainDecorator,
        "multimodal": MultimodalDomainDecorator,
        "multi_modal": MultimodalDomainDecorator,
        "mixed": MultimodalDomainDecorator,
    }

    TASK_MAP: Dict[str, Type[TaskDecorator]] = {
        "classification": ClassificationTaskDecorator,
        "binary_classification": ClassificationTaskDecorator,
        "multi_class_classification": ClassificationTaskDecorator,
        "multiclass_classification": ClassificationTaskDecorator,
        "multi_label_classification": ClassificationTaskDecorator,
        "multilabel_classification": ClassificationTaskDecorator,
        "class_label": ClassificationTaskDecorator,
        "text_classification": ClassificationTaskDecorator,
        "image_classification": ClassificationTaskDecorator,
        "tabular_classification": ClassificationTaskDecorator,
        "audio_classification": ClassificationTaskDecorator,
        "regression": RegressionTaskDecorator,
        "tabular_regression": RegressionTaskDecorator,
        "prediction": RegressionTaskDecorator,
        "continuous": RegressionTaskDecorator,
        "object_detection": ObjectDetectionTaskDecorator,
        "detection": ObjectDetectionTaskDecorator,
        "bounding_box": ObjectDetectionTaskDecorator,
        "boxes": ObjectDetectionTaskDecorator,
        "yolo": ObjectDetectionTaskDecorator,
        "segmentation": SegmentationTaskDecorator,
        "semantic_segmentation": SegmentationTaskDecorator,
        "instance_segmentation": SegmentationTaskDecorator,
        "panoptic_segmentation": SegmentationTaskDecorator,
        "masks": SegmentationTaskDecorator,
        "pixel_classification": SegmentationTaskDecorator,
        "speech_recognition": SpeechRecognitionTaskDecorator,
        "asr": SpeechRecognitionTaskDecorator,
        "automatic_speech_recognition": SpeechRecognitionTaskDecorator,
        "transcription": SpeechRecognitionTaskDecorator,
        "time_series_forecasting": TimeSeriesForecastingTaskDecorator,
        "forecasting": TimeSeriesForecastingTaskDecorator,
        "timeseries_forecasting": TimeSeriesForecastingTaskDecorator,
        "temporal_prediction": TimeSeriesForecastingTaskDecorator,
        "pairwise_classification": PairwiseClassificationTaskDecorator,
        "pairwise": PairwiseClassificationTaskDecorator,
        "matching": PairwiseClassificationTaskDecorator,
        "similarity": PairwiseClassificationTaskDecorator,
        "contrastive": PairwiseClassificationTaskDecorator,
        "seq2seq": Seq2SeqTaskDecorator,
        "sequence_to_sequence": Seq2SeqTaskDecorator,
        "translation": Seq2SeqTaskDecorator,
        "summarization": Seq2SeqTaskDecorator,
        "text_generation": Seq2SeqTaskDecorator,
        "qa": Seq2SeqTaskDecorator,
        "question_answering": Seq2SeqTaskDecorator,
        "ner": NERTaskDecorator,
        "named_entity_recognition": NERTaskDecorator,
        "token_classification": NERTaskDecorator,
        "entity_recognition": NERTaskDecorator,
        "entity_extraction": NERTaskDecorator,
        "sequence_labeling": NERTaskDecorator,
    }

    def _fuzzy_lookup(self, name: str, alias_map: Dict[str, Type]) -> Optional[Type]:
        normalized = _normalize(name)
        if not normalized:
            return None

        if normalized in alias_map:
            return alias_map[normalized]

        matches = difflib.get_close_matches(normalized, alias_map.keys(), n=1, cutoff=0.6)
        if matches:
            matched_key = matches[0]
            logger.info(f"Fuzzy matched '{name}' -> '{matched_key}'")
            return alias_map[matched_key]

        for alias_key, cls in alias_map.items():
            if alias_key in normalized or normalized in alias_key:
                logger.info(f"Substring matched '{name}' -> '{alias_key}'")
                return cls

        return None

    def resolve_domain(self, name: str) -> Optional[DomainDecorator]:
        cls = self._fuzzy_lookup(name, self.DOMAIN_MAP)
        if cls:
            return cls()
        logger.warning(f"Could not resolve domain decorator for: '{name}'")
        return None

    def resolve_task(self, name: str) -> Optional[TaskDecorator]:
        cls = self._fuzzy_lookup(name, self.TASK_MAP)
        if cls:
            return cls()
        logger.warning(f"Could not resolve task decorator for: '{name}'")
        return None

    def list_domains(self):
        seen = set()
        result = []
        for cls in self.DOMAIN_MAP.values():
            if cls not in seen:
                seen.add(cls)
                result.append(cls.__name__)
        return result

    def list_tasks(self):
        seen = set()
        result = []
        for cls in self.TASK_MAP.values():
            if cls not in seen:
                seen.add(cls)
                result.append(cls.__name__)
        return result
