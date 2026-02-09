from .base_decorator import DomainDecorator


class TabularDomainDecorator(DomainDecorator):

    def get_guideline_section(self) -> str:
        return """### DOMAIN: Tabular / Structured Data
- Data consists of rows and columns (CSV, Parquet, or database tables).
- Identify feature types: numerical (continuous/discrete), categorical (nominal/ordinal), datetime, text-like columns.
- Plan explicit handling for each type: imputation strategy per type, encoding strategy for categoricals, scaling for numerics.
- Consider feature engineering: interaction terms, polynomial features, binning, target encoding for high-cardinality categoricals.
- For tree-based models (XGBoost, LightGBM, CatBoost): native categorical support may be available; ordinal encoding often suffices.
- For linear/NN models: one-hot or target encoding for categoricals, StandardScaler or RobustScaler for numerics.
- Check for data leakage: ensure no target-derived features leak into training, fit transformers on train only.
- Handle class imbalance with SMOTE, class weights, or stratified splitting when applicable."""

    def get_preprocessing_section(self) -> str:
        return """### DOMAIN: Tabular Data Preprocessing
- Load data using pandas from CSV/Parquet files.
- Handle missing values: median/mean for numerics, mode or a sentinel for categoricals, consider indicator columns for missingness patterns.
- Encode categoricals: LabelEncoder or OrdinalEncoder for tree models; OneHotEncoder or TargetEncoder for linear/NN models.
- Scale numerics: StandardScaler, MinMaxScaler, or RobustScaler. Fit on training data only, transform both train and test.
- Engineer features: datetime decomposition (year, month, day, dayofweek), interaction features, ratio features.
- Drop constant or near-constant columns, and columns with extreme missing rates (>90%).
- Split data: train/validation with stratification for classification tasks. Use random_state=42.
- Return preprocessed DataFrames/arrays (X_train, X_val, X_test, y_train, y_val)."""

    def get_modeling_section(self) -> str:
        return """### DOMAIN: Tabular Data Modeling
- For traditional ML: XGBoost, LightGBM, CatBoost are strong baselines for tabular data.
- Use early stopping on validation set to prevent overfitting.
- For neural networks on tabular data: consider TabNet, FT-Transformer, or simple MLP with BatchNorm and Dropout.
- Hyperparameter tuning: use optuna or similar for efficient search; limit tuning time.
- Feature importance analysis: use built-in feature importance or SHAP values.
- Ensure all features are numeric before feeding to models (after encoding).
- Use pandas DataFrames or numpy arrays directly with scikit-learn compatible APIs."""

    def get_batch_config(self) -> tuple:
        return (
            "IMPORTANT: Load entire dataset into memory for tabular ML algorithms.",
            "5. Create a function `preprocess_data()` that takes a dictionary of file paths and returns a tuple of **preprocessed DataFrames/arrays** (e.g., X_train, X_val, X_test, y_train, y_val, y_test).",
        )

    def get_data_handling_section(self) -> str:
        return """## TABULAR DATA HANDLING
The preprocessing function returns **preprocessed DataFrames/arrays** (e.g., X_train, X_val, X_test, y_train, y_val, y_test) loaded into memory.
- For traditional ML: use the preprocessed DataFrames/arrays directly with scikit-learn models.
- Logic: Call `X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file_paths)` and use them directly.
- Memory: Data is already loaded into memory and ready for training."""

    def get_assembler_section(self) -> str:
        return """### DOMAIN: Tabular Assembly Notes
- Ensure proper handling of categorical variables for tree-based models.
- Verify that train/test preprocessing is consistent (same encoders, same column order)."""


class CVDomainDecorator(DomainDecorator):

    def get_guideline_section(self) -> str:
        return """### DOMAIN: Computer Vision / Image Data
- Data consists of image files (JPEG, PNG, BMP, TIFF, etc.) organized in folders or referenced by a CSV manifest.
- Image preprocessing: resize to uniform dimensions, normalize pixel values (ImageNet mean/std or dataset-specific), apply augmentation (flips, rotations, color jitter, cutout).
- For traditional ML on images: extract features via a pretrained CNN backbone (EfficientNet, ResNet) in batches to avoid OOM, then train traditional model on extracted feature vectors.
- For neural networks: use torchvision transforms, DataLoader with batch_size, GPU acceleration.
- For pretrained models: use timm, torchvision, or HuggingFace ViT models; match input resolution and normalization to the pretrained model's training configuration.
- **CRITICAL MEMORY CONSTRAINT**: NEVER load all images into a single numpy array. Always use batch processing with generators or DataLoaders.
- Consider data augmentation libraries: albumentations, torchvision.transforms, imgaug.
- For multi-label or multi-class: ensure label encoding matches model output (sigmoid for multi-label, softmax for multi-class)."""

    def get_preprocessing_section(self) -> str:
        return """### DOMAIN: Image Data Preprocessing
- Load image paths/IDs first, NOT the actual images. Use a manifest CSV or scan the image directory.
- Use batch-by-batch processing to prevent memory overflow:
  * For traditional ML feature extraction: process images in batches (batch_size=32-128), extract CNN features per batch, append to a list, clear images from memory. After all batches, concatenate features.
  * For neural network training: use PyTorch DataLoader or tf.data.Dataset with batch_size. Define transforms (resize, normalize, augment) as part of the Dataset class.
- Image transforms: Resize to model's expected input size (e.g., 224x224, 384x384). Normalize with appropriate mean/std. Apply augmentation only to training set.
- Handle variable image sizes by resizing or padding uniformly.
- For pretrained models: match the preprocessing pipeline exactly (e.g., torchvision.transforms for timm/torchvision models, AutoImageProcessor for HuggingFace ViT).
- DO NOT load all images into a single numpy array before feature extraction."""

    def get_modeling_section(self) -> str:
        return """### DOMAIN: Image Data Modeling
- For traditional ML: use features extracted from a pretrained CNN (EfficientNetB0/B3, ResNet50). Train XGBoost/LightGBM on the extracted feature vectors.
- For custom NN: design CNN architectures with Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout, and fully connected layers. Consider residual connections for deeper networks.
- For pretrained models: fine-tune with a lower learning rate (1e-4 to 2e-5). Freeze early layers, unfreeze later layers progressively. Replace the classification head to match the number of target classes.
- Use torchvision.models or timm library for pretrained backbones.
- Training: use Adam/AdamW optimizer, CosineAnnealingLR or OneCycleLR scheduler, early stopping based on validation metric.
- For inference: apply TTA (Test-Time Augmentation) if accuracy is critical.
- Use GPU (torch.device('cuda')) when available for significant speedup."""

    def get_batch_config(self) -> tuple:
        return (
            "IMPORTANT: Use batch-by-batch processing for image data to prevent memory overflow. Use PyTorch DataLoader or process images in batches.",
            "5. Create a function `preprocess_data()` that returns DataLoaders or generators for batch processing of images.",
        )

    def get_data_handling_section(self) -> str:
        return """## IMAGE DATA HANDLING
- Use PyTorch DataLoader with appropriate batch_size (32-128 depending on image size and GPU memory).
- Define a custom Dataset class that loads and transforms images on-the-fly.
- DataLoaders have FINITE length. Training loop iterates through all batches per epoch.
- For traditional ML feature extraction: iterate through DataLoader, extract features with pretrained model, collect into numpy arrays."""

    def get_assembler_section(self) -> str:
        return """### DOMAIN: Image Assembly Notes
- Verify image transforms are applied consistently to train and test data (augmentation only for train).
- Ensure batch processing is maintained throughout the pipeline.
- Check that GPU memory is properly managed (clear cache between phases if needed)."""


class NLPDomainDecorator(DomainDecorator):

    def get_guideline_section(self) -> str:
        return """### DOMAIN: Natural Language Processing / Text Data
- Data consists of text documents, sentences, or token sequences. May be in CSV columns, JSON, or plain text files.
- Text preprocessing: lowercasing, removing special characters, tokenization, stopword removal (for traditional ML); tokenizer-specific preprocessing for transformer models.
- For traditional ML: use TF-IDF, CountVectorizer, or pretrained word embeddings (Word2Vec, GloVe, FastText) to create feature vectors, then train standard classifiers.
- For neural networks: use pretrained tokenizers (HuggingFace AutoTokenizer), pad/truncate sequences to max_length, create attention masks.
- For pretrained models: use HuggingFace transformers (BERT, RoBERTa, DeBERTa, etc.) with PyTorch backend. Fine-tune with appropriate learning rate (2e-5 to 5e-5).
- Handle vocabulary: for traditional ML, limit vocab size; for transformers, the tokenizer handles this automatically.
- Consider text-specific augmentation: back-translation, synonym replacement, random insertion/deletion.
- DO NOT USE NLTK. Use spaCy, HuggingFace tokenizers, or regex-based preprocessing instead."""

    def get_preprocessing_section(self) -> str:
        return """### DOMAIN: Text Data Preprocessing
- Load text data from CSV columns or text files.
- For traditional ML: apply TF-IDF or CountVectorizer. Limit max_features to control dimensionality. Use n-grams (1,2) for better representation.
- For transformer models: use AutoTokenizer.from_pretrained() for model-specific tokenization. Set max_length based on data distribution (128, 256, or 512 tokens). Pad and truncate consistently.
- Create PyTorch Dataset/DataLoader for transformer fine-tuning with input_ids, attention_mask, and labels.
- Handle multilingual text: use multilingual models (mBERT, XLM-R) if language is not English.
- Clean text: remove HTML tags, excessive whitespace, special characters (as appropriate for the task).
- DO NOT USE NLTK for tokenization or preprocessing."""

    def get_modeling_section(self) -> str:
        return """### DOMAIN: Text Data Modeling
- For traditional ML: Logistic Regression, SVM, or XGBoost on TF-IDF/embedding features are strong baselines.
- For custom NN: LSTM, GRU, or 1D-CNN on word embeddings. Use nn.Embedding with pretrained weights (GloVe, FastText).
- For pretrained models: fine-tune HuggingFace transformers (BERT, RoBERTa, DeBERTa). Use AutoModelForSequenceClassification or AutoModelForTokenClassification.
- Training: use AdamW optimizer with linear warmup and decay schedule. Typical learning rates: 2e-5 to 5e-5 for fine-tuning, higher for custom NN.
- Gradient accumulation for effective larger batch sizes when GPU memory is limited.
- Set ignore_mismatched_sizes=True when loading pretrained models with different head sizes.
- Use mixed precision (fp16) for faster training on compatible GPUs."""

    def get_batch_config(self) -> tuple:
        return (
            "IMPORTANT: Use batch processing for text data. Use PyTorch DataLoader with appropriate batch_size for transformer models.",
            "5. Create a function `preprocess_data()` that returns tokenized datasets or DataLoaders suitable for the chosen model.",
        )

    def get_data_handling_section(self) -> str:
        return """## TEXT DATA HANDLING
- For transformer models: use HuggingFace Dataset objects or PyTorch DataLoader with tokenized inputs.
- DataLoaders return batches of input_ids, attention_mask, and labels.
- Iterate through DataLoader for training; FINITE length per epoch.
- For traditional ML: use sparse matrices from TF-IDF vectorizer directly with scikit-learn models."""


class AudioDomainDecorator(DomainDecorator):

    def get_guideline_section(self) -> str:
        return """### DOMAIN: Audio Data
- Data consists of audio files (WAV, MP3, FLAC, OGG) or pre-extracted features.
- Audio preprocessing: resampling to a standard rate (16kHz or 22050Hz), mono conversion, trimming silence, normalization.
- Feature extraction: Mel spectrograms, MFCCs, chroma features, spectral contrast using librosa or torchaudio.
- For traditional ML: extract fixed-length feature vectors (mean/std of MFCCs across time, chroma features) then train standard classifiers.
- For neural networks: use Mel spectrograms as 2D inputs to CNN models, or use raw waveform models (Wav2Vec2, HuBERT).
- For pretrained models: use HuggingFace Wav2Vec2, HuBERT, Whisper models with appropriate feature extractors.
- **CRITICAL MEMORY**: Process audio files in batches. Do not load all audio into memory simultaneously.
- Handle variable-length audio: pad/truncate to fixed length, or use masking."""

    def get_preprocessing_section(self) -> str:
        return """### DOMAIN: Audio Data Preprocessing
- Load audio files using librosa, soundfile, or torchaudio.
- Resample all audio to a consistent sample rate (16000 Hz for speech models, 22050 Hz for music/general).
- Convert stereo to mono if needed.
- Extract features in batches:
  * For traditional ML: extract MFCCs (n_mfcc=40), compute mean/std across time dimension to get fixed-length vectors.
  * For NN: compute Mel spectrograms (n_mels=128, hop_length=512) as 2D arrays.
  * For pretrained models: use AutoFeatureExtractor from HuggingFace for model-specific preprocessing.
- Normalize features: per-sample normalization or dataset-level normalization.
- Handle variable-length audio: pad to max_length or truncate. Use attention masks for padded regions.
- Apply augmentation for training: time stretching, pitch shifting, adding noise, SpecAugment for spectrograms."""

    def get_modeling_section(self) -> str:
        return """### DOMAIN: Audio Data Modeling
- For traditional ML: use MFCC/chroma feature vectors with SVM, XGBoost, or Random Forest.
- For custom NN: use 2D CNN on Mel spectrograms (treat as single-channel images), or 1D CNN on raw waveforms.
- For pretrained models: fine-tune Wav2Vec2, HuBERT, or Whisper from HuggingFace. Use AutoModelForAudioClassification or AutoModelForCTC.
- Training: use AdamW optimizer, CosineAnnealingLR scheduler, early stopping.
- Use torchaudio.transforms for on-the-fly feature extraction in DataLoader.
- For speech recognition: use CTC loss or encoder-decoder architecture with appropriate decoder."""

    def get_batch_config(self) -> tuple:
        return (
            "IMPORTANT: Process audio files in batches to prevent memory overflow. Use DataLoader with appropriate batch_size.",
            "5. Create a function `preprocess_data()` that returns DataLoaders or feature arrays processed in batches.",
        )


class VideoDomainDecorator(DomainDecorator):

    def get_guideline_section(self) -> str:
        return """### DOMAIN: Video Data
- Data consists of video files (MP4, AVI, MOV) or frame sequences.
- Video preprocessing: frame extraction at fixed FPS, resize frames, temporal sampling (uniform or random).
- Feature extraction approaches: per-frame CNN features with temporal pooling, 3D CNN (C3D, I3D, SlowFast), or video transformers (TimeSformer, ViViT).
- For traditional ML: extract frame-level CNN features, aggregate across frames (mean/max pooling), train standard classifier.
- **CRITICAL MEMORY**: Videos are extremely memory-intensive. Process frame-by-frame or in short clips. Never load entire videos into memory.
- Consider temporal dimension: video tasks often require understanding motion and temporal relationships."""

    def get_preprocessing_section(self) -> str:
        return """### DOMAIN: Video Data Preprocessing
- Extract frames using OpenCV (cv2.VideoCapture) at a fixed FPS (e.g., 1-5 FPS for efficiency).
- Resize frames to target resolution (224x224 or 112x112 for video models).
- Temporal sampling: select N frames uniformly from each video (e.g., 16 or 32 frames).
- Normalize frames with ImageNet mean/std or model-specific normalization.
- Create clips (frame sequences) as input tensors of shape (C, T, H, W) or (T, C, H, W).
- Use PyTorch DataLoader with custom Dataset that loads and processes videos on-the-fly.
- NEVER load all video frames into memory at once. Process one video at a time or in small batches."""

    def get_modeling_section(self) -> str:
        return """### DOMAIN: Video Data Modeling
- For traditional ML: extract per-frame features with pretrained CNN, aggregate with temporal pooling, train standard classifier.
- For custom NN: use 3D CNN (Conv3d layers), (2+1)D convolutions, or LSTM/GRU on frame features.
- For pretrained models: use torchvision video models, timm video transformers, or HuggingFace VideoMAE/TimeSformer.
- Training: use smaller batch sizes (4-16) due to memory constraints. Use gradient accumulation for effective larger batches.
- Consider two-stream approaches: spatial (RGB) + temporal (optical flow) streams."""

    def get_batch_config(self) -> tuple:
        return (
            "IMPORTANT: Process video data frame-by-frame or in short clips. NEVER load entire videos into memory. Use very small batch sizes (4-16).",
            "5. Create a function `preprocess_data()` that returns DataLoaders with frame/clip-level batching.",
        )


class TimeSeriesDomainDecorator(DomainDecorator):

    def get_guideline_section(self) -> str:
        return """### DOMAIN: Time Series Data
- Data consists of sequential observations indexed by time (timestamps, dates, or ordered indices).
- Time series preprocessing: handle datetime parsing, sort by time, resample to regular intervals, handle missing timesteps.
- Feature engineering: lag features, rolling statistics (mean, std, min, max over windows), datetime decomposition, trend/seasonality extraction.
- Validation strategy: use time-based splits (walk-forward validation, expanding window). NEVER use random splits for time series as it causes data leakage.
- For traditional ML: engineer lag/rolling features, then train gradient boosting models (LightGBM, XGBoost).
- For neural networks: LSTM, GRU, Temporal Convolutional Networks (TCN), or Transformer-based models (Informer, PatchTST).
- For pretrained models: consider TimesFM, Chronos, or Lag-Llama for zero-shot or fine-tuned forecasting.
- Handle stationarity: differencing, log transforms, detrending if needed for certain models."""

    def get_preprocessing_section(self) -> str:
        return """### DOMAIN: Time Series Data Preprocessing
- Parse datetime columns and set as index. Sort by time.
- Handle missing values: forward fill, interpolation, or model-based imputation. Do NOT use future values to fill past.
- Create lag features: target_lag_1, target_lag_7, target_lag_30 (depending on seasonality).
- Create rolling features: rolling_mean_7, rolling_std_7, rolling_min_7, rolling_max_7.
- Extract datetime features: hour, dayofweek, month, quarter, is_weekend, is_holiday.
- For NN models: create sliding windows of fixed length as input sequences. Shape: (batch, seq_len, features).
- Split data temporally: train on earlier period, validate on later period. NO random shuffling.
- Scale features: fit scaler on training period only, transform validation/test with the same scaler."""

    def get_modeling_section(self) -> str:
        return """### DOMAIN: Time Series Data Modeling
- For traditional ML: use feature-engineered tabular data with LightGBM/XGBoost. Treat as regression/classification on enriched features.
- For custom NN: use LSTM/GRU with sequence inputs, or Temporal Convolutional Networks. Add attention mechanisms for longer sequences.
- For pretrained models: use TimesFM, Chronos, or domain-specific pretrained transformers.
- Evaluation: use time-aware metrics. For forecasting: MAE, RMSE, MAPE. For classification: standard metrics but with temporal split.
- Walk-forward validation: retrain or update model as new data becomes available.
- Handle multiple time series: global model (train on all series) vs local model (one per series)."""

    def get_batch_config(self) -> tuple:
        return (
            "IMPORTANT: Create sliding window sequences for time series data. Ensure temporal ordering is preserved in batches.",
            "5. Create a function `preprocess_data()` that returns time-ordered DataFrames/arrays or sequence DataLoaders with no temporal leakage.",
        )


class MultimodalDomainDecorator(DomainDecorator):

    def get_guideline_section(self) -> str:
        return """### DOMAIN: Multimodal Data (Mixed Modalities)
- Dataset contains multiple data types: e.g., tabular metadata + images, text + images, tabular + audio, etc.
- Design SEPARATE preprocessing pipelines for each modality, then combine features or predictions.
- Fusion strategies: early fusion (concatenate features), late fusion (combine predictions), or cross-attention fusion.
- For traditional ML: extract features from each modality independently (CNN features for images, TF-IDF for text, tabular features), concatenate, train a single model.
- For neural networks: use modality-specific encoders, fuse representations before the prediction head.
- **CRITICAL**: Process each modality with its appropriate tools (pandas for tabular, PIL/torchvision for images, tokenizers for text, librosa for audio).
- Memory management: process non-tabular modalities in batches, keep tabular data in-memory."""

    def get_preprocessing_section(self) -> str:
        return """### DOMAIN: Multimodal Data Preprocessing
- Identify each modality present in the dataset (tabular columns, image paths, text columns, audio paths).
- Preprocess each modality independently:
  * Tabular: standard encoding, scaling, imputation.
  * Images: resize, normalize, batch process with DataLoader.
  * Text: tokenize with appropriate tokenizer.
  * Audio: resample, extract features in batches.
- Align data: ensure each sample has corresponding entries across all modalities (handle missing modalities gracefully).
- For feature fusion: extract features from each modality, concatenate into a unified feature vector.
- For end-to-end fusion: create a custom Dataset that returns all modality inputs per sample."""

    def get_modeling_section(self) -> str:
        return """### DOMAIN: Multimodal Data Modeling
- For traditional ML: concatenate extracted features from all modalities, train a single gradient boosting model.
- For custom NN: design modality-specific encoders (CNN for images, LSTM/Transformer for text, MLP for tabular), concatenate encodings, pass through shared prediction layers.
- For pretrained models: use modality-specific pretrained encoders, combine with a fusion module.
- Consider attention-based fusion to learn the relative importance of each modality.
- Training: balance loss contributions from different modalities if using multi-task learning."""

    def get_batch_config(self) -> tuple:
        return (
            "IMPORTANT: Use batch processing for non-tabular modalities (images, audio, text). Tabular data can be loaded fully into memory. Combine modality features per batch.",
            "5. Create a function `preprocess_data()` that returns combined DataLoaders or separate feature sets per modality for fusion.",
        )
