# 💾 PERSON 2: DATA ENGINEER - Complete Task Guide
## 2-Day Implementation Plan

**Your Role:** Build ALL data pipelines, MLOps infrastructure, and monitoring systems

**Total Time:** 16-20 hours per day
**Deliverables:** 3 dataset loaders + feature extraction + drift monitoring + experiment tracking

---

## 📋 Quick Task Overview

### DAY 1 (16-18 hours)
- **Hours 1-4:** All 3 dataset loaders (Turbofan, MIMII, MVTec)
- **Hours 5-8:** Feature extraction + data augmentation
- **Hours 9-12:** Experiment tracking setup (W&B/MLflow)
- **Hours 13-18:** Evaluation pipeline + metrics tracking

### DAY 2 (16-18 hours)
- **Hours 1-6:** Complete drift detection system
- **Hours 7-12:** Model registry + DVC setup
- **Hours 13-18:** Testing + data validation + final integration

---

## 🚀 DAY 1: DATA INFRASTRUCTURE

### HOUR 1-4: Dataset Loaders (ALL 3 DATASETS)

#### Step 1: Download Datasets First

Create: `scripts/download_datasets.sh`

```bash
#!/bin/bash

echo "======================================"
echo "Downloading all datasets..."
echo "======================================"

# Create directories
mkdir -p data/raw/nasa_turbofan
mkdir -p data/raw/mimii
mkdir -p data/raw/mvtec
mkdir -p data/processed
mkdir -p data/features
mkdir -p models

# NASA Turbofan Dataset
echo ""
echo "1. Downloading NASA Turbofan dataset..."
cd data/raw/nasa_turbofan

# Download from NASA
wget -O CMAPSSData.zip "https://ti.arc.nasa.gov/c/6/"

# If above fails, use alternative mirror
if [ ! -f CMAPSSData.zip ]; then
    echo "Primary source failed, trying alternative..."
    curl -L "https://data.nasa.gov/download/nb9y-zjms/application%2Fzip" -o CMAPSSData.zip
fi

unzip -o CMAPSSData.zip
cd ../../..

echo "✅ Turbofan dataset downloaded"

# MIMII Sound Dataset
echo ""
echo "2. Downloading MIMII sound dataset..."
cd data/raw/mimii

# Download fan data (6dB SNR)
wget https://zenodo.org/record/3384388/files/6_dB_fan.zip

# Extract
unzip -o 6_dB_fan.zip

cd ../../..

echo "✅ MIMII dataset downloaded"

# MVTec AD Dataset
echo ""
echo "3. Downloading MVTec Anomaly Detection dataset..."
cd data/raw/mvtec

# Download (this is large ~4.5GB)
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz

# Extract
tar -xf mvtec_anomaly_detection.tar.xz

cd ../../..

echo "✅ MVTec dataset downloaded"

echo ""
echo "======================================"
echo "All datasets downloaded successfully!"
echo "======================================"
echo ""
echo "Dataset locations:"
echo "  - Turbofan: data/raw/nasa_turbofan/"
echo "  - MIMII: data/raw/mimii/"
echo "  - MVTec: data/raw/mvtec/"
```

**Make executable and run:**
```bash
chmod +x scripts/download_datasets.sh
./scripts/download_datasets.sh
```

---

#### Step 2: NASA Turbofan Loader

Create: `src/data/loaders.py`

```python
"""Data loaders for all datasets"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import torch
from torch.utils.data import Dataset

class TurbofanDataset(Dataset):
    """NASA Turbofan Engine Degradation Dataset Loader

    Dataset Info:
    - Sensors: 21 sensors monitoring engine health
    - Units: Multiple engine units
    - Cycles: Time-series of sensor readings per unit
    - Task: Predict remaining useful life (RUL)

    File format: space-separated values
    Columns: unit_id, cycle, op_setting_1-3, sensor_1-21
    """

    COLUMNS = ['unit', 'cycle'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]

    # Sensors to drop (constant values, no useful info)
    DROP_SENSORS = [1, 5, 6, 10, 16, 18, 19]

    def __init__(self,
                 data_path: Path,
                 sequence_length: int = 50,
                 stride: int = 10,
                 normalize: bool = True,
                 remove_constant_sensors: bool = True):
        """
        Args:
            data_path: Path to train/test file
            sequence_length: Length of time-series sequences
            stride: Stride for sliding window
            normalize: Whether to normalize sensors
            remove_constant_sensors: Remove sensors with no variance
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.remove_constant_sensors = remove_constant_sensors

        # Load and preprocess data
        self.df = self._load_data()
        self.sequences, self.labels = self._create_sequences()

        print(f"✅ Loaded {len(self)} sequences from {data_path.name}")
        print(f"   Shape: {self.sequences.shape}")

    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess raw data"""

        # Read data
        df = pd.read_csv(self.data_path, sep=r'\s+', header=None)

        # Handle extra columns (sometimes there are trailing spaces)
        df = df.iloc[:, :len(self.COLUMNS)]
        df.columns = self.COLUMNS

        # Calculate RUL (Remaining Useful Life)
        # RUL = max_cycle - current_cycle for each unit
        df = df.sort_values(['unit', 'cycle'])
        df['RUL'] = df.groupby('unit')['cycle'].transform('max') - df['cycle']

        # Remove constant sensors
        if self.remove_constant_sensors:
            drop_cols = [f'sensor_{i}' for i in self.DROP_SENSORS]
            df = df.drop(columns=drop_cols, errors='ignore')

        return df

    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences"""

        sequences = []
        labels = []

        # Process each engine unit separately
        for unit_id in self.df['unit'].unique():
            unit_data = self.df[self.df['unit'] == unit_id]

            # Select sensor columns
            sensor_cols = [c for c in unit_data.columns if c.startswith('sensor_')]
            values = unit_data[sensor_cols].values  # [time, sensors]
            rul_values = unit_data['RUL'].values

            # Normalize per-unit if requested
            if self.normalize:
                mean = values.mean(axis=0, keepdims=True)
                std = values.std(axis=0, keepdims=True) + 1e-8
                values = (values - mean) / std

            # Create sliding windows
            for i in range(0, len(values) - self.sequence_length, self.stride):
                seq = values[i:i + self.sequence_length]  # [seq_len, sensors]
                seq = seq.T  # Transpose to [sensors, seq_len] for Conv1D

                # Label: 1 if RUL < 30 (approaching failure), else 0 (normal)
                label = 1 if rul_values[i + self.sequence_length - 1] < 30 else 0

                sequences.append(seq)
                labels.append(label)

        return (
            np.array(sequences, dtype=np.float32),
            np.array(labels, dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sequence

        Returns:
            sequence: [channels, seq_len] tensor
            label: 0 (normal) or 1 (anomaly)
        """
        return (
            torch.from_numpy(self.sequences[idx]),
            torch.tensor(self.labels[idx])
        )

    def get_num_sensors(self) -> int:
        """Get number of sensors (channels)"""
        return self.sequences.shape[1]


class MIMIIDataset(Dataset):
    """MIMII Industrial Sound Dataset

    Dataset Info:
    - Sound recordings from industrial machines
    - Categories: fan, pump, valve, slide rail
    - Normal and anomalous sounds
    - Useful for audio-based anomaly detection

    Note: This implementation uses spectrograms (time-frequency representation)
    """

    def __init__(self,
                 data_path: Path,
                 machine_type: str = 'fan',
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 hop_length: int = 512,
                 n_mels: int = 128):
        """
        Args:
            data_path: Path to MIMII data directory
            machine_type: fan, pump, valve, or slide_rail
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
        """
        self.data_path = Path(data_path) / machine_type
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Find all audio files
        self.audio_files = sorted(list(self.data_path.glob('**/*.wav')))
        self.labels = [0 if 'normal' in str(f) else 1 for f in self.audio_files]

        print(f"✅ Loaded {len(self)} audio files")
        print(f"   Normal: {self.labels.count(0)}")
        print(f"   Anomaly: {self.labels.count(1)}")

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load audio and convert to mel-spectrogram"""
        import librosa

        # Load audio
        audio, sr = librosa.load(self.audio_files[idx], sr=self.sample_rate)

        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        # Convert to tensor
        mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)  # [1, n_mels, time]

        return mel_spec_tensor, torch.tensor(self.labels[idx])
```

---

#### Step 3: MVTec Vision Loader

Create: `src/data/vision_loaders.py`

```python
"""Vision dataset loaders"""

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torch
from typing import Tuple, List

class MVTecDataset(Dataset):
    """MVTec Anomaly Detection Dataset

    Dataset Info:
    - High-resolution images of industrial objects
    - 15 categories (bottle, cable, capsule, etc.)
    - Normal images (training)
    - Anomalous images with pixel-level annotations (testing)

    Categories:
    - Textures: carpet, grid, leather, tile, wood
    - Objects: bottle, cable, capsule, hazelnut, metal_nut,
               pill, screw, toothbrush, transistor, zipper
    """

    CATEGORIES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    def __init__(self,
                 root_dir: str,
                 category: str = 'bottle',
                 split: str = 'train',
                 image_size: int = 256,
                 augment: bool = False):
        """
        Args:
            root_dir: Path to MVTec dataset root
            category: One of CATEGORIES
            split: 'train' or 'test'
            image_size: Resize images to this size
            augment: Whether to apply data augmentation
        """
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.image_size = image_size

        # Build path
        self.data_dir = self.root_dir / category / split

        # Get all images
        self.image_paths = sorted(list(self.data_dir.glob('**/*.png')))

        # Labels: 0 = normal (good), 1 = anomaly
        self.labels = [0 if 'good' in str(p) else 1 for p in self.image_paths]

        # Transforms
        if split == 'train' and augment:
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_test_transform()

        print(f"✅ MVTec {category} ({split}): {len(self)} images")
        print(f"   Normal: {self.labels.count(0)}")
        print(f"   Anomaly: {self.labels.count(1)}")

    def _get_train_transform(self):
        """Training transforms with augmentation"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _get_test_transform(self):
        """Test transforms (no augmentation)"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and transform image"""
        img_path = self.image_paths[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        image = self.transform(image)

        label = self.labels[idx]

        return image, torch.tensor(label)
```

---

#### Step 4: Multimodal Dataset Combiner

Create: `src/data/multimodal_loader.py`

```python
"""Multimodal dataset combining time-series and vision"""

from torch.utils.data import Dataset
import torch
from typing import Tuple

class MultimodalDataset(Dataset):
    """Combine time-series and vision datasets

    Note: Assumes datasets are aligned (same number of samples)
    For real scenarios, you might need more sophisticated alignment
    """

    def __init__(self,
                 timeseries_dataset: Dataset,
                 vision_dataset: Dataset,
                 aligned: bool = True):
        """
        Args:
            timeseries_dataset: Time-series dataset
            vision_dataset: Vision dataset
            aligned: Whether datasets are already aligned
        """
        self.ts_dataset = timeseries_dataset
        self.vis_dataset = vision_dataset
        self.aligned = aligned

        # For simplicity, use minimum length
        self.length = min(len(timeseries_dataset), len(vision_dataset))

        print(f"✅ Multimodal dataset created: {self.length} samples")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get paired time-series and vision sample

        Returns:
            ts_data: Time-series tensor
            vis_data: Vision tensor
            label: Combined label (anomaly if either modality is anomalous)
        """
        ts_data, ts_label = self.ts_dataset[idx]
        vis_data, vis_label = self.vis_dataset[idx]

        # Combined label: anomaly if either is anomalous
        combined_label = max(ts_label, vis_label)

        return ts_data, vis_data, torch.tensor(combined_label)
```

---

### HOUR 5-8: Feature Extraction & Augmentation

#### Create: `src/data/feature_extractors.py`

```python
"""Feature extraction for time-series and audio data"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import pywt
from typing import Dict, Tuple

class SignalFeatureExtractor:
    """Extract frequency and time-domain features from signals"""

    @staticmethod
    def compute_fft(x: np.ndarray, sampling_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Fast Fourier Transform

        Args:
            x: Input signal [time]
            sampling_rate: Sampling rate

        Returns:
            freqs: Frequency bins
            magnitudes: FFT magnitudes
        """
        N = len(x)
        yf = fft(x)
        xf = fftfreq(N, 1 / sampling_rate)[:N//2]
        magnitudes = 2.0/N * np.abs(yf[0:N//2])

        return xf, magnitudes

    @staticmethod
    def compute_spectrogram(x: np.ndarray,
                           fs: float = 1.0,
                           nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram

        Args:
            x: Input signal
            fs: Sampling frequency
            nperseg: Length of each segment

        Returns:
            f: Frequency bins
            t: Time bins
            Sxx: Spectrogram
        """
        f, t, Sxx = signal.spectrogram(x, fs, nperseg=nperseg)
        return f, t, Sxx

    @staticmethod
    def compute_wavelet_transform(x: np.ndarray,
                                  wavelet: str = 'db4',
                                  level: int = 4) -> list:
        """Compute discrete wavelet transform

        Args:
            x: Input signal
            wavelet: Wavelet type (db4, haar, sym, etc.)
            level: Decomposition level

        Returns:
            coeffs: List of wavelet coefficients
        """
        coeffs = pywt.wavedec(x, wavelet, level=level)
        return coeffs

    @staticmethod
    def extract_statistical_features(x: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from signal

        Features:
        - Mean, std, min, max
        - Skewness, kurtosis
        - RMS (Root Mean Square)
        - Peak-to-peak
        - Crest factor
        - Shape factor
        """
        from scipy.stats import skew, kurtosis

        rms = np.sqrt(np.mean(x**2))
        peak_to_peak = np.ptp(x)
        crest_factor = np.max(np.abs(x)) / (rms + 1e-8)

        features = {
            'mean': float(np.mean(x)),
            'std': float(np.std(x)),
            'min': float(np.min(x)),
            'max': float(np.max(x)),
            'median': float(np.median(x)),
            'skew': float(skew(x)),
            'kurtosis': float(kurtosis(x)),
            'rms': float(rms),
            'peak_to_peak': float(peak_to_peak),
            'crest_factor': float(crest_factor),
            'energy': float(np.sum(x**2)),
            'zero_crossing_rate': float(np.sum(np.diff(np.sign(x)) != 0) / len(x))
        }

        return features

    @staticmethod
    def extract_frequency_features(x: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
        """Extract frequency domain features"""

        # FFT
        freqs, mags = SignalFeatureExtractor.compute_fft(x, fs)

        # Power spectral density
        psd = mags ** 2

        # Dominant frequency
        dominant_freq = freqs[np.argmax(psd)]

        # Spectral centroid
        spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-8)

        # Spectral spread
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / (np.sum(psd) + 1e-8))

        # Spectral entropy
        psd_norm = psd / (np.sum(psd) + 1e-8)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-8))

        return {
            'dominant_frequency': float(dominant_freq),
            'spectral_centroid': float(spectral_centroid),
            'spectral_spread': float(spectral_spread),
            'spectral_entropy': float(spectral_entropy),
            'total_power': float(np.sum(psd))
        }
```

#### Create: `src/data/augmentations.py`

```python
"""Data augmentation for time-series"""

import torch
import numpy as np
from typing import Optional

class TimeSeriesAugmentation:
    """Data augmentation techniques for time-series

    Techniques:
    1. Gaussian noise injection
    2. Time warping
    3. Magnitude scaling
    4. Window slicing
    5. Channel dropout
    """

    @staticmethod
    def add_noise(x: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise

        Args:
            x: Input tensor [channels, time] or [batch, channels, time]
            noise_level: Standard deviation of noise
        """
        noise = torch.randn_like(x) * noise_level
        return x + noise

    @staticmethod
    def magnitude_scale(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        """Random magnitude scaling

        Args:
            x: Input tensor
            sigma: Standard deviation of scaling factor
        """
        if x.dim() == 2:  # [channels, time]
            scale = torch.normal(1.0, sigma, size=(x.shape[0], 1))
        else:  # [batch, channels, time]
            scale = torch.normal(1.0, sigma, size=(x.shape[0], x.shape[1], 1))

        return x * scale

    @staticmethod
    def time_warp(x: torch.Tensor, sigma: float = 0.2) -> torch.Tensor:
        """Random time warping using cubic interpolation

        Args:
            x: Input tensor [channels, time]
            sigma: Standard deviation of warping
        """
        from scipy.interpolate import CubicSpline

        if x.dim() == 3:
            # Batch processing
            batch_size = x.shape[0]
            warped = []
            for i in range(batch_size):
                warped.append(TimeSeriesAugmentation.time_warp(x[i], sigma))
            return torch.stack(warped)

        # Single sample
        orig_steps = np.arange(x.shape[-1])

        # Generate random warping curve
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[-1],))
        warp_steps = np.cumsum(random_warps)
        warp_steps = (warp_steps - warp_steps[0]) / (warp_steps[-1] - warp_steps[0]) * (x.shape[-1] - 1)

        # Interpolate
        warped = np.zeros_like(x.numpy())
        for i in range(x.shape[0]):  # For each channel
            cs = CubicSpline(warp_steps, x[i].numpy())
            warped[i] = cs(orig_steps)

        return torch.from_numpy(warped).float()

    @staticmethod
    def window_slice(x: torch.Tensor, reduce_ratio: float = 0.9) -> torch.Tensor:
        """Random window slicing

        Args:
            x: Input tensor
            reduce_ratio: Ratio of window to keep
        """
        target_len = int(x.shape[-1] * reduce_ratio)
        start_idx = np.random.randint(0, x.shape[-1] - target_len + 1)

        return x[..., start_idx:start_idx + target_len]

    @staticmethod
    def channel_dropout(x: torch.Tensor, dropout_prob: float = 0.1) -> torch.Tensor:
        """Randomly drop channels (set to zero)

        Args:
            x: Input tensor [channels, time]
            dropout_prob: Probability of dropping each channel
        """
        mask = torch.rand(x.shape[0], 1) > dropout_prob
        return x * mask

    @staticmethod
    def mixup(x1: torch.Tensor, x2: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
        """Mixup augmentation

        Args:
            x1, x2: Two input tensors
            alpha: Mixup parameter
        """
        lam = np.random.beta(alpha, alpha)
        return lam * x1 + (1 - lam) * x2


class VisionAugmentation:
    """Data augmentation for images (already handled by torchvision in loaders)"""

    @staticmethod
    def cutout(image: torch.Tensor, n_holes: int = 1, length: int = 16) -> torch.Tensor:
        """Cutout augmentation

        Args:
            image: Input image [C, H, W]
            n_holes: Number of holes
            length: Length of hole
        """
        h, w = image.shape[1:]
        mask = torch.ones_like(image)

        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[:, y1:y2, x1:x2] = 0

        return image * mask
```

---

### HOUR 9-12: Experiment Tracking & MLOps

#### Create: `src/utils/experiment_tracker.py`

```python
"""Experiment tracking with W&B and MLflow"""

import wandb
import mlflow
from typing import Dict, Any, Optional
from pathlib import Path

class ExperimentTracker:
    """Unified interface for W&B and MLflow

    Usage:
        tracker = ExperimentTracker(backend='wandb', project='my-project')
        tracker.log_params({'lr': 0.001})
        tracker.log_metrics({'loss': 0.5}, step=100)
        tracker.log_artifact('model.pth')
    """

    def __init__(self,
                 backend: str = 'wandb',
                 project: str = 'vae-anomaly-detection',
                 experiment_name: Optional[str] = None,
                 config: Optional[Dict] = None):
        """
        Args:
            backend: 'wandb' or 'mlflow'
            project: Project name
            experiment_name: Name of this experiment/run
            config: Initial configuration dictionary
        """
        self.backend = backend

        if backend == 'wandb':
            self.run = wandb.init(
                project=project,
                name=experiment_name,
                config=config or {}
            )
        elif backend == 'mlflow':
            mlflow.set_experiment(project)
            mlflow.start_run(run_name=experiment_name)
            if config:
                mlflow.log_params(config)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        if self.backend == 'wandb':
            wandb.config.update(params)
        elif self.backend == 'mlflow':
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if self.backend == 'wandb':
            wandb.log(metrics, step=step)
        elif self.backend == 'mlflow':
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)

    def log_artifact(self, file_path: str, artifact_type: str = 'model'):
        """Log artifact (file)"""
        if self.backend == 'wandb':
            wandb.save(file_path)
        elif self.backend == 'mlflow':
            mlflow.log_artifact(file_path, artifact_type)

    def log_model(self, model, model_name: str = 'model'):
        """Log PyTorch model"""
        if self.backend == 'wandb':
            torch.save(model.state_dict(), f'{model_name}.pth')
            wandb.save(f'{model_name}.pth')
        elif self.backend == 'mlflow':
            mlflow.pytorch.log_model(model, model_name)

    def finish(self):
        """Finish tracking"""
        if self.backend == 'wandb':
            wandb.finish()
        elif self.backend == 'mlflow':
            mlflow.end_run()
```

#### Create: `src/utils/model_registry.py`

```python
"""Model registry with DVC"""

import dvc.api
from pathlib import Path
import shutil
import json
from typing import Dict, Optional
import hashlib

class ModelRegistry:
    """Model versioning and registry with DVC

    Features:
    - Version models with DVC
    - Track model metadata
    - Easy model retrieval
    """

    def __init__(self, registry_dir: str = 'models/registry'):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.registry_dir / 'metadata.json'
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load metadata file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save metadata file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_hash(self, file_path: Path) -> str:
        """Compute file hash"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def register_model(self,
                      model_path: str,
                      model_name: str,
                      version: str,
                      metrics: Dict[str, float],
                      tags: Optional[Dict[str, str]] = None):
        """Register a model

        Args:
            model_path: Path to model checkpoint
            model_name: Name of model (e.g., 'timeseries_vae')
            version: Version string (e.g., 'v1.0')
            metrics: Performance metrics
            tags: Additional metadata tags
        """
        model_path = Path(model_path)

        # Copy to registry
        dest_path = self.registry_dir / f'{model_name}_{version}.ckpt'
        shutil.copy(model_path, dest_path)

        # Compute hash
        file_hash = self._compute_hash(dest_path)

        # Store metadata
        model_id = f'{model_name}_{version}'
        self.metadata[model_id] = {
            'name': model_name,
            'version': version,
            'path': str(dest_path),
            'hash': file_hash,
            'metrics': metrics,
            'tags': tags or {}
        }

        self._save_metadata()

        print(f"✅ Model registered: {model_id}")
        print(f"   Path: {dest_path}")
        print(f"   Metrics: {metrics}")

        return model_id

    def get_model(self, model_name: str, version: str = 'latest') -> Dict:
        """Retrieve model info"""
        if version == 'latest':
            # Find latest version
            versions = [k for k in self.metadata.keys() if k.startswith(model_name)]
            if not versions:
                raise ValueError(f"No model found with name: {model_name}")
            model_id = sorted(versions)[-1]
        else:
            model_id = f'{model_name}_{version}'

        if model_id not in self.metadata:
            raise ValueError(f"Model not found: {model_id}")

        return self.metadata[model_id]

    def list_models(self) -> Dict:
        """List all registered models"""
        return self.metadata
```

---

### HOUR 13-18: Evaluation Pipeline & Metrics

#### Create: `src/monitoring/metrics.py`

```python
"""Production metrics tracking"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path

@dataclass
class ProductionMetrics:
    """Metrics for a single prediction batch"""
    timestamp: str
    num_predictions: int
    num_anomalies: int
    avg_score: float
    max_score: float
    min_score: float
    latency_ms: float
    drift_score: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class MetricsTracker:
    """Track production metrics over time"""

    def __init__(self, storage_path: str = 'data/metrics/production_metrics.jsonl'):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.metrics_history: List[ProductionMetrics] = []
        self._load_history()

    def _load_history(self):
        """Load existing metrics history"""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.metrics_history.append(ProductionMetrics(**data))

    def log_batch(self,
                  scores: np.ndarray,
                  latency_ms: float,
                  drift_score: float = 0.0,
                  threshold: float = 0.5):
        """Log metrics for a batch of predictions

        Args:
            scores: Anomaly scores
            latency_ms: Inference latency in milliseconds
            drift_score: Drift detection score
            threshold: Anomaly threshold
        """
        metrics = ProductionMetrics(
            timestamp=datetime.now().isoformat(),
            num_predictions=len(scores),
            num_anomalies=int((scores > threshold).sum()),
            avg_score=float(scores.mean()),
            max_score=float(scores.max()),
            min_score=float(scores.min()),
            latency_ms=latency_ms,
            drift_score=drift_score
        )

        self.metrics_history.append(metrics)

        # Append to file
        with open(self.storage_path, 'a') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')

    def get_summary(self, window_hours: int = 24) -> Dict:
        """Get summary statistics for recent window

        Args:
            window_hours: Time window in hours

        Returns:
            Dictionary with aggregated metrics
        """
        cutoff = datetime.now() - timedelta(hours=window_hours)
        cutoff_str = cutoff.isoformat()

        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_str
        ]

        if not recent_metrics:
            return {}

        total_preds = sum(m.num_predictions for m in recent_metrics)
        total_anomalies = sum(m.num_anomalies for m in recent_metrics)

        return {
            'window_hours': window_hours,
            'total_predictions': total_preds,
            'total_anomalies': total_anomalies,
            'anomaly_rate': total_anomalies / total_preds if total_preds > 0 else 0,
            'avg_latency_ms': np.mean([m.latency_ms for m in recent_metrics]),
            'max_latency_ms': max(m.latency_ms for m in recent_metrics),
            'avg_score': np.mean([m.avg_score for m in recent_metrics]),
            'max_drift_score': max(m.drift_score for m in recent_metrics),
            'num_batches': len(recent_metrics)
        }

    def export_to_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        summary = self.get_summary(window_hours=1)

        prometheus_metrics = f"""
# HELP vae_predictions_total Total number of predictions
# TYPE vae_predictions_total counter
vae_predictions_total {summary.get('total_predictions', 0)}

# HELP vae_anomalies_total Total number of anomalies detected
# TYPE vae_anomalies_total counter
vae_anomalies_total {summary.get('total_anomalies', 0)}

# HELP vae_latency_ms Inference latency in milliseconds
# TYPE vae_latency_ms gauge
vae_latency_ms {{quantile="0.5"}} {summary.get('avg_latency_ms', 0)}
vae_latency_ms {{quantile="1.0"}} {summary.get('max_latency_ms', 0)}

# HELP vae_drift_score Drift detection score
# TYPE vae_drift_score gauge
vae_drift_score {summary.get('max_drift_score', 0)}
        """

        return prometheus_metrics.strip()
```

---

## 🚀 DAY 2: DRIFT MONITORING & TESTING

### HOUR 1-6: Complete Drift Detection System

#### Create: `src/monitoring/drift_detector.py`

```python
"""Distribution drift detection for production monitoring"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import pickle
from pathlib import Path

class DriftDetector:
    """Detect distribution drift in production

    Methods:
    - Maximum Mean Discrepancy (MMD)
    - Kolmogorov-Smirnov test
    - Wasserstein distance
    - Chi-squared test
    """

    def __init__(self, reference_window_size: int = 1000):
        self.reference_window_size = reference_window_size
        self.reference_latents: Optional[np.ndarray] = None
        self.reference_scores: Optional[np.ndarray] = None

    def set_reference(self, latents: np.ndarray, scores: np.ndarray):
        """Set reference distribution from training/validation data

        Args:
            latents: Latent representations [N, latent_dim]
            scores: Anomaly scores [N]
        """
        self.reference_latents = latents
        self.reference_scores = scores

        print(f"✅ Reference distribution set:")
        print(f"   Latents shape: {latents.shape}")
        print(f"   Scores shape: {scores.shape}")

    def detect_latent_drift(self,
                           current_latents: np.ndarray,
                           method: str = 'mmd',
                           **kwargs) -> Dict[str, float]:
        """Detect drift in latent space

        Args:
            current_latents: Current latent representations
            method: 'mmd', 'ks', or 'wasserstein'

        Returns:
            Dictionary with drift score and details
        """
        if self.reference_latents is None:
            raise ValueError("Reference distribution not set. Call set_reference() first.")

        if method == 'mmd':
            drift_score = self._maximum_mean_discrepancy(
                self.reference_latents,
                current_latents,
                gamma=kwargs.get('gamma', 1.0)
            )
        elif method == 'ks':
            drift_score = self._kolmogorov_smirnov(
                self.reference_latents,
                current_latents
            )
        elif method == 'wasserstein':
            drift_score = self._wasserstein_distance(
                self.reference_latents,
                current_latents
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            'drift_score': drift_score,
            'method': method,
            'drift_detected': self._is_drift_significant(drift_score, method)
        }

    @staticmethod
    def _maximum_mean_discrepancy(X: np.ndarray,
                                  Y: np.ndarray,
                                  gamma: float = 1.0) -> float:
        """Compute Maximum Mean Discrepancy (MMD) between two distributions

        MMD is a kernel-based metric that measures distribution similarity

        Args:
            X, Y: Two sets of samples [N, D]
            gamma: RBF kernel bandwidth parameter

        Returns:
            MMD score (0 = identical, higher = more different)
        """
        # Compute kernel matrices
        XX = np.dot(X, X.T)
        YY = np.dot(Y, Y.T)
        XY = np.dot(X, Y.T)

        # RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2)
        X_diag = XX.diagonal().reshape(-1, 1)
        Y_diag = YY.diagonal().reshape(-1, 1)

        K_XX = np.exp(-gamma * (X_diag + X_diag.T - 2 * XX))
        K_YY = np.exp(-gamma * (Y_diag + Y_diag.T - 2 * YY))
        K_XY = np.exp(-gamma * (X_diag + Y_diag.T - 2 * XY))

        # MMD = E[K(X,X)] + E[K(Y,Y)] - 2*E[K(X,Y)]
        mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

        return float(max(0, mmd))  # Clip to non-negative

    @staticmethod
    def _kolmogorov_smirnov(X: np.ndarray, Y: np.ndarray) -> float:
        """Kolmogorov-Smirnov test (multi-dimensional)

        Performs KS test on each dimension and returns max statistic

        Args:
            X, Y: Two sets of samples [N, D]

        Returns:
            Max KS statistic across all dimensions
        """
        ks_stats = []

        for dim in range(X.shape[1]):
            stat, pval = stats.ks_2samp(X[:, dim], Y[:, dim])
            ks_stats.append(stat)

        return float(np.max(ks_stats))

    @staticmethod
    def _wasserstein_distance(X: np.ndarray, Y: np.ndarray) -> float:
        """Wasserstein distance (Earth Mover's Distance)

        Averaged over all dimensions

        Args:
            X, Y: Two sets of samples [N, D]

        Returns:
            Average Wasserstein distance
        """
        distances = []

        for dim in range(X.shape[1]):
            dist = stats.wasserstein_distance(X[:, dim], Y[:, dim])
            distances.append(dist)

        return float(np.mean(distances))

    def detect_score_drift(self, current_scores: np.ndarray) -> Dict[str, float]:
        """Detect drift in anomaly score distribution

        Args:
            current_scores: Current anomaly scores

        Returns:
            Dictionary with drift statistics
        """
        if self.reference_scores is None:
            raise ValueError("Reference scores not set. Call set_reference() first.")

        # KS test
        ks_stat, ks_pval = stats.ks_2samp(self.reference_scores, current_scores)

        # Mean shift (in standard deviations)
        ref_mean = np.mean(self.reference_scores)
        ref_std = np.std(self.reference_scores)
        cur_mean = np.mean(current_scores)
        mean_shift = (cur_mean - ref_mean) / (ref_std + 1e-8)

        # Variance ratio
        ref_var = np.var(self.reference_scores)
        cur_var = np.var(current_scores)
        var_ratio = cur_var / (ref_var + 1e-8)

        # Drift detection
        drift_detected = (ks_pval < 0.05) or (abs(mean_shift) > 2) or (var_ratio > 2 or var_ratio < 0.5)

        return {
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval),
            'mean_shift_sigma': float(mean_shift),
            'variance_ratio': float(var_ratio),
            'drift_detected': drift_detected,
            'ref_mean': float(ref_mean),
            'cur_mean': float(cur_mean),
            'ref_std': float(ref_std),
            'cur_std': float(np.std(current_scores))
        }

    @staticmethod
    def _is_drift_significant(drift_score: float, method: str) -> bool:
        """Determine if drift is significant based on method-specific thresholds"""
        thresholds = {
            'mmd': 0.1,
            'ks': 0.2,
            'wasserstein': 0.3
        }
        return drift_score > thresholds.get(method, 0.1)

    def should_retrain(self, drift_results: Dict) -> Tuple[bool, str]:
        """Determine if model should be retrained based on drift

        Args:
            drift_results: Results from drift detection

        Returns:
            (should_retrain, reason)
        """
        reasons = []

        # Check latent drift
        if 'drift_score' in drift_results:
            if drift_results.get('drift_detected', False):
                reasons.append(f"Latent drift detected: {drift_results['drift_score']:.4f}")

        # Check score drift
        if 'mean_shift_sigma' in drift_results:
            if abs(drift_results['mean_shift_sigma']) > 3:
                reasons.append(f"Large mean shift: {drift_results['mean_shift_sigma']:.2f}σ")

        if 'variance_ratio' in drift_results:
            var_ratio = drift_results['variance_ratio']
            if var_ratio > 2 or var_ratio < 0.5:
                reasons.append(f"Variance change: {var_ratio:.2f}x")

        if 'ks_pvalue' in drift_results:
            if drift_results['ks_pvalue'] < 0.01:
                reasons.append(f"KS test significant: p={drift_results['ks_pvalue']:.4f}")

        should_retrain = len(reasons) > 0
        reason_str = "; ".join(reasons) if reasons else "No significant drift"

        return should_retrain, reason_str

    def save(self, path: str):
        """Save drift detector state"""
        state = {
            'reference_latents': self.reference_latents,
            'reference_scores': self.reference_scores,
            'reference_window_size': self.reference_window_size
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

        print(f"✅ Drift detector saved to {path}")

    def load(self, path: str):
        """Load drift detector state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.reference_latents = state['reference_latents']
        self.reference_scores = state['reference_scores']
        self.reference_window_size = state.get('reference_window_size', 1000)

        print(f"✅ Drift detector loaded from {path}")
```

---

### HOUR 7-12: DVC Setup & Data Validation

#### Create: `.dvc/config`

```yaml
[core]
    remote = storage
    autostage = true

['remote "storage"']
    url = data/dvc-storage
```

#### Create: `scripts/setup_dvc.sh`

```bash
#!/bin/bash

echo "Setting up DVC..."

# Initialize DVC
dvc init

# Add data to DVC
echo "Adding datasets to DVC..."
dvc add data/raw/nasa_turbofan
dvc add data/raw/mimii
dvc add data/raw/mvtec

# Add models
dvc add models/

# Commit
git add data/.gitignore data/raw/.dvc models/.gitignore models.dvc .dvc/config
git commit -m "Setup DVC for data and model versioning"

echo "✅ DVC setup complete"
```

#### Create: `src/data/data_validation.py`

```python
"""Data validation with Great Expectations"""

import great_expectations as ge
from great_expectations.dataset import PandasDataset
import pandas as pd
from typing import Dict, List

class DataValidator:
    """Validate data quality using Great Expectations"""

    def __init__(self):
        self.expectations = []

    def validate_turbofan_data(self, df: pd.DataFrame) -> Dict:
        """Validate NASA Turbofan dataset"""

        gdf = ge.from_pandas(df)

        # Column existence
        gdf.expect_column_to_exist('unit')
        gdf.expect_column_to_exist('cycle')

        # Value ranges
        gdf.expect_column_values_to_be_between('unit', min_value=1, max_value=None)
        gdf.expect_column_values_to_be_between('cycle', min_value=1, max_value=None)

        # No nulls in critical columns
        gdf.expect_column_values_to_not_be_null('unit')
        gdf.expect_column_values_to_not_be_null('cycle')

        # Sensor value ranges (reasonable physical bounds)
        sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
        for col in sensor_cols:
            gdf.expect_column_values_to_be_between(col, min_value=-100, max_value=100, mostly=0.95)

        # Get results
        results = gdf.validate()

        return {
            'success': results.success,
            'num_validations': len(results.results),
            'num_successful': sum(r.success for r in results.results),
            'num_failed': sum(not r.success for r in results.results)
        }

    def validate_mvtec_dataset(self, image_paths: List[str]) -> Dict:
        """Validate MVTec dataset"""

        validations = {
            'total_images': len(image_paths),
            'valid_images': 0,
            'invalid_images': 0,
            'errors': []
        }

        from PIL import Image

        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                img.verify()
                validations['valid_images'] += 1
            except Exception as e:
                validations['invalid_images'] += 1
                validations['errors'].append(str(e))

        validations['success'] = validations['invalid_images'] == 0

        return validations
```

---

### HOUR 13-18: Testing & Integration

#### Create: `tests/unit/test_loaders.py`

```python
"""Unit tests for data loaders"""

import pytest
import torch
from pathlib import Path

def test_turbofan_dataset():
    """Test NASA Turbofan dataset loader"""
    from src.data.loaders import TurbofanDataset

    dataset = TurbofanDataset(
        data_path='data/raw/nasa_turbofan/train_FD001.txt',
        sequence_length=50,
        stride=10
    )

    assert len(dataset) > 0

    # Test single sample
    sample, label = dataset[0]
    assert sample.shape == (14, 50)  # [channels, time]
    assert label in [0, 1]

    # Test batch
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    batch, labels = next(iter(loader))
    assert batch.shape == (32, 14, 50)


def test_mvtec_dataset():
    """Test MVTec dataset loader"""
    from src.data.vision_loaders import MVTecDataset

    dataset = MVTecDataset(
        root_dir='data/raw/mvtec',
        category='bottle',
        split='train'
    )

    assert len(dataset) > 0

    sample, label = dataset[0]
    assert sample.shape == (3, 256, 256)  # [C, H, W]


def test_multimodal_dataset():
    """Test multimodal dataset"""
    from src.data.multimodal_loader import MultimodalDataset
    from src.data.loaders import TurbofanDataset
    from src.data.vision_loaders import MVTecDataset

    ts_dataset = TurbofanDataset('data/raw/nasa_turbofan/train_FD001.txt')
    vis_dataset = MVTecDataset('data/raw/mvtec', category='bottle')

    mm_dataset = MultimodalDataset(ts_dataset, vis_dataset)

    ts_data, vis_data, label = mm_dataset[0]
    assert ts_data.shape[0] == 14
    assert vis_data.shape == (3, 256, 256)
```

#### Create: `tests/unit/test_drift_detector.py`

```python
"""Unit tests for drift detection"""

import numpy as np
import pytest

def test_drift_detector_mmd():
    """Test MMD drift detection"""
    from src.monitoring.drift_detector import DriftDetector

    detector = DriftDetector()

    # Create reference and current distributions
    ref_data = np.random.randn(1000, 10)
    cur_data = np.random.randn(1000, 10)  # Same distribution

    detector.set_reference(ref_data, np.random.rand(1000))

    # No drift expected
    result = detector.detect_latent_drift(cur_data, method='mmd')
    assert result['drift_score'] < 0.1

    # Create drifted data
    drifted_data = np.random.randn(1000, 10) + 2.0  # Mean shift

    result_drift = detector.detect_latent_drift(drifted_data, method='mmd')
    assert result_drift['drift_score'] > result['drift_score']


def test_drift_detector_ks():
    """Test KS drift detection"""
    from src.monitoring.drift_detector import DriftDetector

    detector = DriftDetector()

    ref_data = np.random.randn(1000, 5)
    detector.set_reference(ref_data, np.random.rand(1000))

    # Test same distribution
    result = detector.detect_latent_drift(np.random.randn(1000, 5), method='ks')
    assert result['drift_score'] < 0.3

    # Test different distribution
    result_drift = detector.detect_latent_drift(np.random.randn(1000, 5) * 2, method='ks')
    assert result_drift['drift_score'] > result['drift_score']
```

---

## ✅ Your Deliverables Checklist

### Data Pipeline
- [ ] NASA Turbofan loader working
- [ ] MIMII audio loader working
- [ ] MVTec vision loader working
- [ ] Multimodal dataset combiner working
- [ ] All datasets downloaded and validated

### Feature Engineering
- [ ] FFT feature extraction
- [ ] Wavelet transforms
- [ ] Statistical features
- [ ] Time-series augmentation
- [ ] Vision augmentation

### MLOps
- [ ] Experiment tracking (W&B/MLflow) setup
- [ ] Model registry with DVC
- [ ] Metrics tracking system
- [ ] Data validation pipeline

### Drift Monitoring
- [ ] MMD drift detection
- [ ] KS test implementation
- [ ] Wasserstein distance
- [ ] Score drift detection
- [ ] Retraining triggers

### Testing
- [ ] Unit tests for all loaders
- [ ] Integration tests
- [ ] Data validation tests
- [ ] Drift detection tests

---

## 🚨 Troubleshooting

**Dataset Download Fails:**
```bash
# Use alternative sources or manual download
# Check network connection
# Verify disk space (need ~10GB total)
```

**DVC Issues:**
```bash
# Reinitialize
dvc destroy
dvc init
```

**Great Expectations Errors:**
```python
# Simplify expectations if needed
# Focus on critical validations only
```

---

## 📞 Sync Points with Team

**After 4 hours:**
- Confirm all datasets downloaded
- Share dataset info with Person 1
- Provide data loaders for training

**End of Day 1:**
- Provide experiment tracking access
- Share evaluation pipeline
- Confirm metrics tracking working

**After 6 hours Day 2:**
- Share drift detection results
- Provide model registry access
- Final testing coordination

---

**You're the backbone of the system! Data quality = Model quality. Let's do this! 💪**
