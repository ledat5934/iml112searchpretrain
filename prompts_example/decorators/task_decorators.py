from .base_decorator import TaskDecorator


class ClassificationTaskDecorator(TaskDecorator):

    def get_guideline_section(self) -> str:
        return """### TASK: Classification
- Objective: predict a discrete class label for each sample.
- Determine if binary or multi-class classification from the number of unique target values.
- Evaluation metrics: Accuracy, F1-score (macro/micro/weighted), AUC-ROC (binary or one-vs-rest), Precision, Recall, Log Loss.
- Handle class imbalance: use class_weight='balanced', SMOTE oversampling, focal loss, or stratified sampling.
- Loss functions: CrossEntropyLoss (multi-class), BCEWithLogitsLoss (binary or multi-label).
- Output layer: softmax for multi-class (mutually exclusive), sigmoid for multi-label (independent labels).
- Ensure stratified train/validation split to preserve class distribution.
- For probabilistic predictions: calibrate probabilities if needed (Platt scaling, isotonic regression)."""

    def get_preprocessing_section(self) -> str:
        return """### TASK: Classification Preprocessing
- Encode target labels: LabelEncoder for string labels, ensure consistent mapping for train/test.
- Use stratified splitting (stratify=y in train_test_split) to preserve class distribution.
- For multi-label: use MultiLabelBinarizer to encode target columns.
- Consider over/under sampling for imbalanced classes during preprocessing."""

    def get_modeling_section(self) -> str:
        return """### TASK: Classification Modeling
- Choose appropriate loss: CrossEntropyLoss for multi-class, BCEWithLogitsLoss for binary/multi-label.
- Set class_weight or sample_weight for imbalanced datasets.
- For traditional ML: use classification-specific models (LogisticRegression, RandomForestClassifier, XGBClassifier).
- Threshold tuning: for binary classification, optimize decision threshold on validation set.
- Print validation metrics: accuracy, F1-score, and the competition-specified metric.
- For submission: output class labels or probabilities as required by submission format."""


class RegressionTaskDecorator(TaskDecorator):

    def get_guideline_section(self) -> str:
        return """### TASK: Regression
- Objective: predict a continuous numerical value for each sample.
- Evaluation metrics: RMSE, MSE, MAE, R-squared (R2), MAPE, Huber loss.
- Consider target transformation: log-transform for skewed targets (log1p/expm1), Box-Cox, or quantile normalization.
- Loss functions: MSELoss (L2), L1Loss (MAE), HuberLoss (robust to outliers), SmoothL1Loss.
- Handle outliers in target: consider clipping extreme values or using robust loss functions.
- For bounded targets: apply appropriate constraints (e.g., clamp predictions to valid range).
- Ensure predictions are inverse-transformed if target was transformed during preprocessing."""

    def get_preprocessing_section(self) -> str:
        return """### TASK: Regression Preprocessing
- Analyze target distribution: check for skewness, outliers, and bounded ranges.
- Apply target transformation if highly skewed: np.log1p() for positive skewed targets.
- No stratification needed for splitting; simple random split is fine.
- Scale target if using neural networks (StandardScaler on y_train, inverse_transform predictions)."""

    def get_modeling_section(self) -> str:
        return """### TASK: Regression Modeling
- For traditional ML: XGBRegressor, LGBMRegressor, CatBoostRegressor, Ridge, Lasso, ElasticNet.
- For neural networks: final layer with 1 output neuron, no activation (or ReLU if target is non-negative).
- Loss: MSELoss is standard; use HuberLoss if outliers are present.
- Print validation metric: RMSE or MAE along with the competition-specified metric.
- Inverse-transform predictions if target was transformed during preprocessing.
- For submission: output continuous values matching the expected format."""


class ObjectDetectionTaskDecorator(TaskDecorator):

    def get_guideline_section(self) -> str:
        return """### TASK: Object Detection
- Objective: detect and localize objects in images with bounding boxes and class labels.
- Annotation formats: COCO JSON (x, y, width, height), YOLO TXT (class cx cy w h normalized), Pascal VOC XML (xmin, ymin, xmax, ymax).
- Evaluation metrics: mAP (mean Average Precision) at various IoU thresholds (mAP@0.5, mAP@0.5:0.95), IoU, Precision, Recall.
- Model families: YOLO (v5, v8, v9, v10), Faster R-CNN, DETR, RT-DETR, EfficientDet.
- For Ultralytics YOLO models: generate a data.yaml configuration file pointing to train/val/test image directories and class names.
- Post-processing: Non-Maximum Suppression (NMS) with appropriate IoU threshold, confidence thresholding.
- Data augmentation: mosaic, mixup, random crop, horizontal flip, HSV augmentation."""

    def get_preprocessing_section(self) -> str:
        return """### TASK: Object Detection Preprocessing
- Convert annotations to the required format (COCO JSON, YOLO TXT, or Pascal VOC XML) based on the chosen model.
- For YOLO: create data.yaml with paths to train/val/test image directories and class names list. Organize images and labels in the expected directory structure.
- For COCO-format models: create annotations JSON with images, annotations, and categories.
- Validate annotations: check for invalid bounding boxes (negative values, boxes outside image bounds), missing annotations.
- Split data: maintain image-level splits (all annotations for one image go to same split).
- No feature scaling needed for image pixels (handled by model's preprocessing)."""

    def get_modeling_section(self) -> str:
        return """### TASK: Object Detection Modeling
- For pretrained models: Ultralytics YOLO/RT-DETR are recommended for ease of use. Use model.train(data='data.yaml', epochs=N).
- For custom models: use torchvision.models.detection (Faster R-CNN, FCOS, RetinaNet) with pretrained backbones.
- For Ultralytics: pass data.yaml path to model.train(), NOT DataLoader directly.
- Training: use appropriate image size (640 for YOLO), batch_size (8-16), learning rate (0.01 default for YOLO).
- Prediction: run model.predict() on test images, parse results to get bounding boxes, class labels, and confidence scores.
- Submission: format predictions according to competition requirements (often COCO-style JSON or CSV with box coordinates)."""


class SegmentationTaskDecorator(TaskDecorator):

    def get_guideline_section(self) -> str:
        return """### TASK: Segmentation (Semantic / Instance)
- Objective: classify each pixel (semantic) or detect individual object instances with pixel-level masks (instance).
- Evaluation metrics: IoU (Intersection over Union), Dice coefficient, pixel accuracy, mean IoU across classes.
- Model families: U-Net, DeepLabV3/V3+, Mask R-CNN (instance), SegFormer, Segment Anything (SAM).
- Loss functions: CrossEntropyLoss (per-pixel), DiceLoss, combined CE+Dice, FocalLoss for class-imbalanced pixels.
- Input: images at appropriate resolution. Output: segmentation masks (H x W) with class indices or binary masks per instance.
- Data augmentation: geometric transforms (flip, rotate, crop) applied identically to image AND mask."""

    def get_preprocessing_section(self) -> str:
        return """### TASK: Segmentation Preprocessing
- Load images and corresponding mask annotations (PNG masks, COCO polygons, or RLE-encoded masks).
- Resize images and masks to uniform dimensions. Use nearest-neighbor interpolation for masks to preserve label values.
- Normalize images (ImageNet mean/std for pretrained encoders).
- Apply augmentation jointly to image and mask: random flip, rotation, crop, elastic transform.
- For semantic segmentation: masks should be (H, W) with integer class labels.
- For instance segmentation: masks should be per-instance binary masks.
- Use PyTorch DataLoader with custom Dataset that returns (image, mask) pairs."""

    def get_modeling_section(self) -> str:
        return """### TASK: Segmentation Modeling
- U-Net variants: U-Net, U-Net++, Attention U-Net. Use segmentation_models_pytorch (smp) library for easy setup.
- DeepLab: DeepLabV3+ with pretrained backbone (ResNet, EfficientNet) from torchvision.
- Instance segmentation: Mask R-CNN from torchvision.models.detection.
- Loss: combine DiceLoss + CrossEntropyLoss for better convergence. Use class weights for imbalanced pixel distributions.
- Training: standard image batch training. Monitor IoU/Dice on validation set.
- Prediction: output per-pixel class predictions (argmax over channels). Apply CRF post-processing if needed.
- Submission: encode masks as specified (RLE encoding, PNG files, COCO JSON format)."""


class SpeechRecognitionTaskDecorator(TaskDecorator):

    def get_guideline_section(self) -> str:
        return """### TASK: Speech Recognition / ASR
- Objective: transcribe audio speech into text (automatic speech recognition).
- Evaluation metrics: WER (Word Error Rate), CER (Character Error Rate).
- Model families: Wav2Vec2, HuBERT, Whisper (OpenAI), Conformer, DeepSpeech.
- Loss functions: CTC (Connectionist Temporal Classification) loss for CTC-based models, CrossEntropyLoss for encoder-decoder models.
- Audio preprocessing: 16kHz sample rate, mono channel, normalize amplitude.
- Decoding: greedy decoding, beam search, or language model-assisted decoding.
- Consider language-specific models for non-English speech."""

    def get_preprocessing_section(self) -> str:
        return """### TASK: Speech Recognition Preprocessing
- Resample audio to 16000 Hz (standard for speech models).
- Use model-specific feature extractors: Wav2Vec2FeatureExtractor, WhisperFeatureExtractor from HuggingFace.
- Tokenize transcriptions using model-specific tokenizer (Wav2Vec2CTCTokenizer, WhisperTokenizer).
- Pad/truncate audio to consistent length or use dynamic batching.
- Create DataLoader with (audio_features, transcript_ids) pairs."""

    def get_modeling_section(self) -> str:
        return """### TASK: Speech Recognition Modeling
- Preferred: fine-tune Whisper or Wav2Vec2 from HuggingFace.
- Use AutoModelForCTC (CTC-based) or AutoModelForSpeechSeq2Seq (encoder-decoder like Whisper).
- Training: use CTC loss or cross-entropy depending on model architecture.
- Decoding: use model.generate() for Whisper-style models, or processor.batch_decode() for CTC models.
- Evaluate WER on validation set using jiwer library or manual computation.
- Submission: output transcribed text per audio sample."""


class TimeSeriesForecastingTaskDecorator(TaskDecorator):

    def get_guideline_section(self) -> str:
        return """### TASK: Time Series Forecasting
- Objective: predict future values of a time-dependent variable given historical observations.
- Evaluation metrics: MAPE, SMAPE, RMSE, MAE, MASE, Quantile Loss (for probabilistic forecasts).
- Validation: use temporal splits only (walk-forward / expanding window). NEVER random split for time series.
- Model families: ARIMA/SARIMA (statistical), LightGBM with lag features (ML), LSTM/GRU (NN), Temporal Fusion Transformer, N-BEATS, PatchTST.
- Forecast horizons: single-step vs multi-step. For multi-step: direct (separate model per step), recursive (iterative), or direct-recursive hybrid.
- Handle seasonality: use seasonal features, seasonal differencing, or Fourier features.
- Handle multiple series: global model (shared parameters across series) vs local model (one per series)."""

    def get_preprocessing_section(self) -> str:
        return """### TASK: Time Series Forecasting Preprocessing
- Sort data by timestamp. Handle irregular timestamps by resampling.
- Create target variable lag features: y(t-1), y(t-2), ..., y(t-k).
- Create rolling window features: rolling mean, std, min, max over various windows.
- Extract calendar features: hour, dayofweek, month, holiday indicators.
- For NN models: create input-output sequence pairs with sliding window. Input: (seq_len, features), Output: (horizon,).
- Split temporally: last N periods as test, second-to-last M periods as validation.
- Scale features and target: fit on training window only."""

    def get_modeling_section(self) -> str:
        return """### TASK: Time Series Forecasting Modeling
- For traditional ML: create feature-engineered dataset with lag/rolling features, train LightGBM/XGBoost regressor.
- For custom NN: LSTM/GRU with sequence input, or TCN (temporal convolution), or Transformer with positional encoding.
- For pretrained: use TimesFM, Chronos, Lag-Llama for zero-shot or fine-tuned forecasting.
- Multi-step forecasting: train separate model per horizon step (direct) or predict recursively.
- Probabilistic forecasting: predict quantiles (10th, 50th, 90th) if required.
- Post-processing: inverse-transform scaled predictions, clip to valid range."""


class PairwiseClassificationTaskDecorator(TaskDecorator):

    def get_guideline_section(self) -> str:
        return """### TASK: Pairwise Classification / Matching
- Objective: determine if two inputs (text pairs, image pairs, entity pairs) are similar, related, or match.
- Evaluation metrics: Accuracy, F1, AUC-ROC, Precision@k, MRR (Mean Reciprocal Rank).
- Approaches: Siamese networks (shared encoder for both inputs), cross-encoder (concatenate inputs), bi-encoder (separate encoders + similarity).
- Loss functions: ContrastiveLoss, TripletLoss, CosineSimilarity + BCE, CrossEntropyLoss on concatenated representations.
- For text: use sentence-transformers library or fine-tune BERT-style cross-encoder.
- For images: Siamese CNN with contrastive or triplet loss.
- Handle negative sampling: hard negative mining for better training signal."""

    def get_preprocessing_section(self) -> str:
        return """### TASK: Pairwise Classification Preprocessing
- Create pairs: (input_a, input_b, label) where label is 1 (match) or 0 (non-match).
- For text pairs: tokenize both texts (cross-encoder: concatenate with [SEP]; bi-encoder: tokenize separately).
- For image pairs: load both images, apply same transforms.
- Balance positive/negative pairs if imbalanced. Consider hard negative mining.
- Create DataLoader that yields (input_a, input_b, label) tuples."""

    def get_modeling_section(self) -> str:
        return """### TASK: Pairwise Classification Modeling
- Cross-encoder: concatenate inputs, pass through shared model, classify with binary head. Higher accuracy but slower inference.
- Bi-encoder: encode each input separately, compute similarity (cosine, dot product), threshold for classification. Faster inference.
- Siamese networks: shared weights encoder for both inputs. Use contrastive loss or triplet loss.
- For text: fine-tune sentence-transformers or BERT cross-encoder.
- For images: ResNet/EfficientNet backbone with Siamese architecture.
- Submission: output match/no-match labels or similarity scores."""


class NERTaskDecorator(TaskDecorator):

    def get_guideline_section(self) -> str:
        return """### TASK: Named Entity Recognition (NER) / Token Classification
- Objective: assign a label to each token in a sequence (e.g., B-PER, I-PER, O, B-ORG, I-ORG).
- Labeling scheme: BIO (Begin-Inside-Outside), BIOES/BILOU, or IOB2. Ensure consistent scheme across train/test.
- Evaluation metrics: entity-level F1-score (strict & partial), precision, recall using seqeval library. Token-level accuracy is secondary.
- Model families: BERT/RoBERTa/DeBERTa + token classification head, BiLSTM-CRF, SpaCy NER, Flair.
- Loss functions: CrossEntropyLoss per token (ignore padding index = -100), CRF layer for structured prediction.
- Subword alignment: when using transformer tokenizers, map word-level labels to subword tokens. Assign label only to the first subword of each word; set -100 for subsequent subwords and special tokens.
- Post-processing: convert token-level predictions back to entity spans (start, end, label). Merge B-/I- tags into contiguous entities.
- For nested NER: consider span-based models or multi-label token classification."""

    def get_preprocessing_section(self) -> str:
        return """### TASK: NER Preprocessing
- Parse annotation format: CoNLL (tab/space-separated columns), JSON spans (start, end, label), or IOB-tagged sequences.
- Tokenize with AutoTokenizer.from_pretrained(). Set is_split_into_words=True when input is pre-tokenized word list.
- Align labels to subword tokens: use tokenizer.word_ids() to map subword positions back to word positions. Assign label only to the first subword of each word; set -100 for padding, special tokens ([CLS], [SEP]), and non-first subwords.
- Build label-to-id and id-to-label mappings from the full tag set (including O, B-*, I-* tags).
- Set max_length based on sentence length distribution (128 or 256 tokens typically suffices).
- Create PyTorch Dataset returning input_ids, attention_mask, and labels (with -100 for ignored positions).
- Use DataCollatorForTokenClassification from HuggingFace for dynamic padding."""

    def get_modeling_section(self) -> str:
        return """### TASK: NER Modeling
- Preferred: fine-tune AutoModelForTokenClassification from HuggingFace (BERT, RoBERTa, DeBERTa).
- Set num_labels to the total number of unique BIO tags. Pass id2label and label2id to model config.
- Loss: the model computes CrossEntropyLoss internally, ignoring tokens with label = -100.
- Optional CRF layer: add a CRF on top of transformer outputs for better structured prediction (use torchcrf or pytorch-crf library).
- Training: AdamW optimizer with linear warmup, lr=2e-5 to 5e-5, weight_decay=0.01. Train for 5-15 epochs with early stopping on validation entity-level F1.
- Evaluation: use seqeval library (classification_report, f1_score with mode='strict', scheme=IOB2). Evaluate at entity level, not token level.
- Prediction: run model on test data, convert logits to predicted tag IDs (argmax), map back to tag strings, then extract entity spans.
- Submission: output entities per sample as required (entity text, label, start/end offsets, or BIO-tagged sequence)."""


class Seq2SeqTaskDecorator(TaskDecorator):

    def get_guideline_section(self) -> str:
        return """### TASK: Sequence-to-Sequence (Seq2Seq)
- Objective: transform an input sequence into an output sequence (translation, summarization, question answering, text generation).
- Evaluation metrics: BLEU (translation), ROUGE (summarization), METEOR, BERTScore, Exact Match (QA).
- Model families: T5, BART, mBART, MarianMT (translation), Pegasus (summarization), GPT-2/GPT-Neo (generation).
- Loss functions: CrossEntropyLoss on decoder output tokens (teacher forcing during training).
- Decoding strategies: greedy, beam search (num_beams=4-8), top-k sampling, nucleus (top-p) sampling.
- For pretrained models: use HuggingFace AutoModelForSeq2SeqLM or AutoModelForCausalLM.
- Handle long sequences: truncate or use models with longer context windows (LongT5, LED)."""

    def get_preprocessing_section(self) -> str:
        return """### TASK: Seq2Seq Preprocessing
- Tokenize both source and target sequences with model-specific tokenizer.
- Set appropriate max_source_length and max_target_length based on data distribution.
- For translation: source = input language text, target = output language text.
- For summarization: source = full document, target = summary.
- Create DataLoader with input_ids, attention_mask, and labels (target token IDs).
- Use HuggingFace DataCollatorForSeq2Seq for dynamic padding and label preparation."""

    def get_modeling_section(self) -> str:
        return """### TASK: Seq2Seq Modeling
- Fine-tune pretrained models: T5 or BART for general seq2seq, MarianMT for translation, Pegasus for summarization.
- Use AutoModelForSeq2SeqLM.from_pretrained() with appropriate model name.
- Training: use AdamW optimizer, linear warmup with decay, label smoothing if needed.
- Generation: use model.generate() with beam search (num_beams=4), max_length, length_penalty.
- Evaluate with sacrebleu (BLEU), rouge_score (ROUGE-L), or bertscore.
- Submission: output generated text per input sample."""
