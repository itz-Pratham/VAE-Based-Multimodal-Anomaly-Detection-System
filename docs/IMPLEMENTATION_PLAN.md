# 🏗️ VAE-Based Multimodal Anomaly Detection System
## 4-Week Production-Grade Implementation Plan

---

## 👥 Team Structure & Roles

**Person 1 (ML Engineer - Model Development):** VAE architecture, training pipeline, model optimization
**Person 2 (Data Engineer - Pipeline & Infrastructure):** Data ingestion, preprocessing, feature engineering, MLOps
**Person 3 (Full-Stack Engineer - Dashboard & Deployment):** Visualization, monitoring dashboards, API development

---

## 📚 Complete Tech Stack

### Core ML Stack
- **Framework:** PyTorch 2.1+ with PyTorch Lightning 2.1+
- **Numerical Computing:** NumPy 1.24+, SciPy 1.11+
- **Data Processing:** Pandas 2.1+, Polars 0.19+ (faster than Pandas)
- **Signal Processing:** librosa 0.10+ (audio), scikit-image 0.22+ (vision)

### MLOps & Experiment Tracking
- **Experiment Tracking:** Weights & Biases (wandb) or MLflow 2.9+
- **Model Versioning:** DVC 3.0+
- **Hyperparameter Tuning:** Optuna 3.4+
- **Model Serving:** FastAPI 0.104+, ONNX Runtime 1.16+

### Data Infrastructure
- **Storage:** MinIO (S3-compatible) or AWS S3
- **Data Validation:** Great Expectations 0.18+
- **Feature Store:** Feast 0.35+ (optional but recommended)
- **Streaming:** Apache Kafka + Kafka-Python (for real-time inference)

### Visualization & Monitoring
- **Dashboard:** Streamlit 1.29+ (primary), Plotly Dash 2.14+ (alternative)
- **Plotting:** Plotly 5.18+, Matplotlib 3.8+, Seaborn 0.13+
- **Monitoring:** Prometheus + Grafana (production metrics)
- **Explainability:** SHAP 0.43+, Captum 0.7+

### Deployment & DevOps
- **Containerization:** Docker 24+, Docker Compose
- **Orchestration:** Kubernetes (optional for production)
- **CI/CD:** GitHub Actions
- **API Gateway:** Nginx or Traefik

### Testing & Quality
- **Testing:** pytest 7.4+, pytest-cov
- **Linting:** ruff 0.1+ (replaces flake8, black, isort)
- **Type Checking:** mypy 1.7+
- **Pre-commit:** pre-commit 3.5+

---

## 🗂️ Project Structure

```
VAE-Anomaly-Detection/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd.yml
│       └── model-training.yml
├── configs/
│   ├── model/
│   │   ├── vae_timeseries.yaml
│   │   ├── vae_vision.yaml
│   │   └── vae_multimodal.yaml
│   ├── data/
│   │   ├── nasa_turbofan.yaml
│   │   ├── mimii.yaml
│   │   └── mvtec.yaml
│   └── training/
│       └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── models/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── preprocessors.py
│   │   ├── augmentations.py
│   │   ├── feature_extractors.py
│   │   ├── vision_loaders.py
│   │   ├── vision_augmentations.py
│   │   └── multimodal_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py
│   │   ├── decoders.py
│   │   ├── vae.py
│   │   ├── vision_vae.py
│   │   ├── multimodal_vae.py
│   │   └── losses.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   └── optimizers.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── anomaly_detector.py
│   │   ├── threshold_calibrator.py
│   │   └── root_cause_analyzer.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift_detector.py
│   │   └── metrics.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logging.py
├── dashboards/
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── 1_training_monitor.py
│   │   ├── 2_anomaly_detection.py
│   │   ├── 3_root_cause.py
│   │   └── 4_drift_monitoring.py
│   └── components/
│       ├── plots.py
│       └── widgets.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_prototyping.ipynb
│   └── 04_evaluation.ipynb
├── tests/
│   ├── unit/
│   │   ├── test_loaders.py
│   │   ├── test_models.py
│   │   ├── test_anomaly_detector.py
│   │   └── test_drift_detector.py
│   ├── integration/
│   │   ├── test_training_pipeline.py
│   │   └── test_inference_pipeline.py
│   └── e2e/
│       └── test_api.py
├── docker/
│   ├── Dockerfile.train
│   ├── Dockerfile.serve
│   ├── Dockerfile.dashboard
│   ├── docker-compose.yml
│   └── nginx.conf
├── scripts/
│   ├── download_datasets.sh
│   ├── train.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   └── deploy.py
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   ├── deployment_guide.md
│   └── model_cards/
│       ├── timeseries_vae.md
│       ├── vision_vae.md
│       └── multimodal_vae.md
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── pyproject.toml
├── setup.py
├── .pre-commit-config.yaml
├── .gitignore
├── IMPLEMENTATION_PLAN.md
└── README.md
```

---

## 📅 WEEK 1: Foundation & Time-Series VAE

### Phase 1A: Infrastructure Setup (Days 1-2)

#### Person 2 (Data Engineer) - LEAD

**Priority Tasks:**
1. ✅ Set up project structure (use structure above)
2. ✅ Create virtual environment and install base dependencies
3. ✅ Configure MLOps infrastructure (Weights & Biases or MLflow)
4. ✅ Implement data download pipeline for NASA Turbofan dataset
5. ✅ Create data validation pipeline with Great Expectations
6. ✅ Set up DVC for data versioning

**Files to Create:**
- `requirements/base.txt`, `requirements/dev.txt`, `requirements/prod.txt`
- `scripts/download_datasets.sh`
- `src/data/loaders.py` - TurbofanDataset class
- `configs/data/nasa_turbofan.yaml`
- `.dvc/config`

**Code Reference:**
```python
# src/data/loaders.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import torch
from torch.utils.data import Dataset

class TurbofanDataset(Dataset):
    """NASA Turbofan Engine Degradation Dataset Loader"""

    COLUMNS = ['unit', 'cycle'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]

    def __init__(self,
                 data_path: Path,
                 sequence_length: int = 50,
                 stride: int = 1,
                 normalize: bool = True):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize

        self.df = self._load_data()
        self.sequences, self.labels = self._create_sequences()

    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess raw data"""
        df = pd.read_csv(self.data_path, sep=' ', header=None)
        df.dropna(axis=1, how='all', inplace=True)
        df.columns = self.COLUMNS

        # Calculate RUL (Remaining Useful Life)
        df = df.sort_values(['unit', 'cycle'])
        df['RUL'] = df.groupby('unit')['cycle'].transform('max') - df['cycle']

        return df

    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences"""
        sequences = []
        labels = []

        for unit_id in self.df['unit'].unique():
            unit_data = self.df[self.df['unit'] == unit_id]

            sensor_cols = [c for c in unit_data.columns if c.startswith('sensor_')]
            values = unit_data[sensor_cols].values
            rul_values = unit_data['RUL'].values

            if self.normalize:
                values = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-8)

            for i in range(0, len(values) - self.sequence_length, self.stride):
                seq = values[i:i + self.sequence_length]
                label = rul_values[i + self.sequence_length - 1]
                sequences.append(seq)
                labels.append(1 if label < 30 else 0)

        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.sequences[idx]),
            torch.tensor(self.labels[idx])
        )
```

```bash
# scripts/download_datasets.sh
#!/bin/bash

echo "Creating data directories..."
mkdir -p data/raw/nasa_turbofan
mkdir -p data/raw/mimii
mkdir -p data/raw/mvtec
mkdir -p data/processed
mkdir -p data/features
mkdir -p models

echo "Downloading NASA Turbofan dataset..."
wget https://ti.arc.nasa.gov/c/6/ -O data/raw/nasa_turbofan/CMAPSSData.zip
unzip data/raw/nasa_turbofan/CMAPSSData.zip -d data/raw/nasa_turbofan/

echo "Downloading MIMII Sound dataset..."
wget https://zenodo.org/record/3384388/files/6_dB_fan.zip -O data/raw/mimii/fan.zip
unzip data/raw/mimii/fan.zip -d data/raw/mimii/

echo "Downloading MVTec AD dataset..."
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xf mvtec_anomaly_detection.tar.xz -C data/raw/mvtec/

echo "Dataset download complete!"
```

**Validation & Testing:**
```python
# tests/unit/test_loaders.py
import pytest
from src.data.loaders import TurbofanDataset

def test_turbofan_dataset_loading():
    dataset = TurbofanDataset('data/raw/nasa_turbofan/train_FD001.txt')
    assert len(dataset) > 0

    sample, label = dataset[0]
    assert sample.shape == (50, 14)  # sequence_length x num_sensors
    assert label in [0, 1]
```

---

#### Person 3 (Full-Stack) - SUPPORT

**Priority Tasks:**
1. ✅ Set up Docker environment
2. ✅ Create initial Streamlit dashboard skeleton
3. ✅ Configure CI/CD with GitHub Actions
4. ✅ Set up pre-commit hooks

**Files to Create:**
- `docker/Dockerfile.train`
- `docker/docker-compose.yml`
- `.github/workflows/ci.yml`
- `dashboards/streamlit_app.py`
- `.pre-commit-config.yaml`

**Code Reference:**
```dockerfile
# docker/Dockerfile.train
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements/base.txt requirements/dev.txt ./
RUN pip install --no-cache-dir -r base.txt -r dev.txt

# Copy source code
COPY . .

# Install package
RUN pip install -e .

CMD ["python", "scripts/train.py"]
```

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements/dev.txt

    - name: Run linting
      run: |
        ruff check src/ tests/

    - name: Run type checking
      run: |
        mypy src/

    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

```python
# dashboards/streamlit_app.py
import streamlit as st

st.set_page_config(
    page_title="VAE Anomaly Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("🏭 Industrial Anomaly Detection Dashboard")
st.sidebar.success("Select a page above")

st.markdown("""
## System Overview
This system implements production-grade multimodal anomaly detection using Variational Autoencoders.

### Implementation Phases
- **Phase 1:** Time-Series VAE (Week 1-2) ⏳
- **Phase 2:** Vision VAE (Week 3) ⏳
- **Phase 3:** Multimodal Fusion (Week 4) ⏳

### Current Status
- ✅ Infrastructure Setup
- ⏳ Model Training
- ⏳ Deployment

### Team Members
- **ML Engineer:** VAE architecture & training
- **Data Engineer:** Pipeline & MLOps
- **Full-Stack Engineer:** Dashboards & API
""")

# System metrics placeholder
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Models Trained", "0", "0")
with col2:
    st.metric("Datasets Loaded", "1/3", "+1")
with col3:
    st.metric("API Endpoints", "0/5", "0")
```

**Deliverable:** Phase 1A Dashboard showing project status

---

#### Person 1 (ML Engineer) - SUPPORT

**Priority Tasks:**
1. ✅ Review literature on VAE-based anomaly detection
2. ✅ Set up experiment tracking (W&B account)
3. ✅ Create data exploration notebook
4. ✅ Design VAE architecture specifications

**Files to Create:**
- `notebooks/01_data_exploration.ipynb`
- `configs/model/vae_timeseries.yaml`
- `docs/architecture.md`

**Architecture Specifications:**
```yaml
# configs/model/vae_timeseries.yaml
model:
  name: "timeseries_vae"
  encoder_type: "cnn"  # Options: cnn, transformer

  encoder:
    input_channels: 14  # Number of sensors
    hidden_dims: [64, 128, 256]
    latent_dim: 128
    dropout: 0.2

  decoder:
    hidden_dims: [256, 128, 64]
    output_channels: 14

  loss:
    beta: 1.0  # KL divergence weight (β-VAE)
    reconstruction_loss: "mse"  # Options: mse, mae

training:
  batch_size: 128
  learning_rate: 0.001
  max_epochs: 100
  early_stopping_patience: 10
  gradient_clip_val: 1.0
```

---

### Phase 1B: Time-Series VAE Development (Days 3-5)

#### Person 1 (ML Engineer) - LEAD

**Priority Tasks:**
1. ✅ Implement 1D-CNN VAE encoder
2. ✅ Implement 1D-CNN VAE decoder
3. ✅ Create VAE PyTorch Lightning module
4. ✅ Implement loss functions (reconstruction + KL divergence)
5. ✅ Create training script with logging
6. ✅ Train initial model and iterate

**Files to Create:**
- `src/models/encoders.py`
- `src/models/decoders.py`
- `src/models/vae.py`
- `src/models/losses.py`
- `scripts/train.py`
- `src/training/trainer.py`
- `src/training/callbacks.py`

**Code Reference:**
```python
# src/models/encoders.py
import torch
import torch.nn as nn
from typing import Tuple

class Conv1DEncoder(nn.Module):
    """1D-CNN Encoder for time-series data"""

    def __init__(self,
                 input_channels: int,
                 sequence_length: int,
                 latent_dim: int = 128,
                 hidden_dims: list = [64, 128, 256]):
        super().__init__()

        self.latent_dim = latent_dim

        modules = []
        in_channels = input_channels

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2)
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.flatten_size = self._get_flatten_size(input_channels, sequence_length)

        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def _get_flatten_size(self, input_channels: int, sequence_length: int) -> int:
        x = torch.zeros(1, input_channels, sequence_length)
        x = self.encoder(x)
        return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
```

```python
# src/models/vae.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Tuple

class TimeSeriesVAE(pl.LightningModule):
    """Variational Autoencoder for Time-Series Anomaly Detection"""

    def __init__(self,
                 input_channels: int,
                 sequence_length: int,
                 latent_dim: int = 128,
                 encoder_type: str = 'cnn',
                 beta: float = 1.0,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.beta = beta
        self.learning_rate = learning_rate

        if encoder_type == 'cnn':
            from .encoders import Conv1DEncoder
            self.encoder = Conv1DEncoder(input_channels, sequence_length, latent_dim)

        from .decoders import Conv1DDecoder
        self.decoder = Conv1DDecoder(latent_dim, input_channels, sequence_length)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def compute_loss(self, x: torch.Tensor, reconstruction: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.beta * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, _ = batch
        reconstruction, mu, logvar = self(x)
        losses = self.compute_loss(x, reconstruction, mu, logvar)

        self.log_dict({f'train/{k}': v for k, v in losses.items()},
                     on_step=True, on_epoch=True, prog_bar=True)

        return losses['loss']

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, labels = batch
        reconstruction, mu, logvar = self(x)
        losses = self.compute_loss(x, reconstruction, mu, logvar)

        self.log_dict({f'val/{k}': v for k, v in losses.items()},
                     on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': losses['loss']}

    def compute_anomaly_score(self, x: torch.Tensor, reconstruction: torch.Tensor,
                               mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        recon_error = torch.mean((x - reconstruction) ** 2, dim=(1, 2))
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return recon_error + self.beta * kl_div

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }
```

```python
# scripts/train.py
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="training/default", version_base=None)
def main(cfg: DictConfig):
    # W&B logger
    wandb_logger = WandbLogger(
        project="vae-anomaly-detection",
        name=f"timeseries-vae-{cfg.model.encoder_type}",
        config=dict(cfg)
    )

    # Data
    from src.data.loaders import TurbofanDataset
    from torch.utils.data import DataLoader, random_split

    full_dataset = TurbofanDataset(
        data_path=cfg.data.path,
        sequence_length=cfg.data.sequence_length,
        stride=cfg.data.stride
    )

    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size,
                            num_workers=4)

    # Model
    from src.models.vae import TimeSeriesVAE

    model = TimeSeriesVAE(
        input_channels=len(cfg.data.sensors.selected),
        sequence_length=cfg.data.sequence_length,
        latent_dim=cfg.model.latent_dim,
        encoder_type=cfg.model.encoder_type,
        beta=cfg.model.beta,
        learning_rate=cfg.training.learning_rate
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"models/checkpoints/{wandb_logger.experiment.id}",
        filename="vae-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3
    )

    early_stop_callback = EarlyStopping(monitor="val/loss", patience=10, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer
    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print(f"Training complete! Best model: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
```

**Run Training:**
```bash
python scripts/train.py
```

---

#### Person 2 (Data Engineer) - SUPPORT

**Priority Tasks:**
1. ✅ Implement feature extraction pipeline (FFT, wavelets, spectrograms)
2. ✅ Create data augmentation module
3. ✅ Monitor training experiments
4. ✅ Create data preprocessing utilities

**Files to Create:**
- `src/data/feature_extractors.py`
- `src/data/augmentations.py`
- `src/data/preprocessors.py`

**Code Reference:**
```python
# src/data/feature_extractors.py
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import pywt

class SignalFeatureExtractor:
    """Extract frequency and time-domain features"""

    @staticmethod
    def compute_fft(x: np.ndarray, sampling_rate: float = 1.0) -> tuple:
        N = len(x)
        yf = fft(x)
        xf = fftfreq(N, 1 / sampling_rate)[:N//2]
        return xf, 2.0/N * np.abs(yf[0:N//2])

    @staticmethod
    def compute_spectrogram(x: np.ndarray, fs: float = 1.0, nperseg: int = 256):
        f, t, Sxx = signal.spectrogram(x, fs, nperseg=nperseg)
        return f, t, Sxx

    @staticmethod
    def compute_wavelet_transform(x: np.ndarray, wavelet: str = 'db4', level: int = 4):
        coeffs = pywt.wavedec(x, wavelet, level=level)
        return coeffs

    @staticmethod
    def extract_statistical_features(x: np.ndarray) -> dict:
        from scipy.stats import skew, kurtosis
        return {
            'mean': np.mean(x),
            'std': np.std(x),
            'skew': skew(x),
            'kurtosis': kurtosis(x),
            'rms': np.sqrt(np.mean(x**2)),
            'peak_to_peak': np.ptp(x),
            'crest_factor': np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-8)
        }
```

```python
# src/data/augmentations.py
import torch
import numpy as np

class TimeSeriesAugmentation:
    """Data augmentation for time-series"""

    @staticmethod
    def add_noise(x: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
        noise = torch.randn_like(x) * noise_level
        return x + noise

    @staticmethod
    def magnitude_scale(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        scale = torch.normal(1.0, sigma, size=(x.shape[0], x.shape[1], 1))
        return x * scale

    @staticmethod
    def time_warp(x: torch.Tensor, sigma: float = 0.2) -> torch.Tensor:
        from scipy.interpolate import CubicSpline

        orig_steps = np.arange(x.shape[-1])
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[-1],))
        warp_steps = np.cumsum(random_warps)
        warp_steps = (warp_steps - warp_steps[0]) / (warp_steps[-1] - warp_steps[0]) * (x.shape[-1] - 1)

        warped = np.zeros_like(x.numpy())
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                cs = CubicSpline(warp_steps, x[i, j].numpy())
                warped[i, j] = cs(orig_steps)

        return torch.from_numpy(warped).float()
```

---

#### Person 3 (Full-Stack) - SUPPORT

**Priority Tasks:**
1. ✅ Create training monitoring dashboard
2. ✅ Implement real-time metric visualization
3. ✅ Set up model checkpoint browser

**Files to Create:**
- `dashboards/pages/1_training_monitor.py`
- `dashboards/components/plots.py`

**Code Reference:**
```python
# dashboards/pages/1_training_monitor.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import wandb
import pandas as pd

st.title("📊 Training Monitor")

# Connect to W&B
api_key = st.sidebar.text_input("W&B API Key", type="password")
if api_key:
    wandb.login(key=api_key)

# Project selection
project = st.sidebar.text_input("Project Name", "vae-anomaly-detection")

if project:
    api = wandb.Api()
    runs = api.runs(f"your-entity/{project}")

    run_names = [run.name for run in runs]
    selected_run = st.selectbox("Select Training Run", run_names)

    if selected_run:
        run = [r for r in runs if r.name == selected_run][0]
        history = run.history()

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Train Loss", f"{history['train/loss'].iloc[-1]:.4f}")
        with col2:
            st.metric("Final Val Loss", f"{history['val/loss'].iloc[-1]:.4f}")
        with col3:
            st.metric("Best Val Loss", f"{history['val/loss'].min():.4f}")

        # Loss curves
        st.subheader("Loss Curves")
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history['train/loss'], name='Train Loss', mode='lines'))
            fig.add_trace(go.Scatter(y=history['val/loss'], name='Val Loss', mode='lines'))
            fig.update_layout(title="Training & Validation Loss", xaxis_title="Step", yaxis_title="Loss")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history['train/recon_loss'], name='Recon Loss'))
            fig.add_trace(go.Scatter(y=history['train/kl_loss'], name='KL Loss'))
            fig.update_layout(title="Loss Components", xaxis_title="Step", yaxis_title="Loss")
            st.plotly_chart(fig, use_container_width=True)

        # Hyperparameters
        st.subheader("Hyperparameters")
        st.json(run.config)
```

---

### Week 1 Deliverables

**Completed by End of Week 1:**
- ✅ Complete project infrastructure
- ✅ Data pipeline for NASA Turbofan dataset
- ✅ Time-series VAE (CNN-based) implemented and trained
- ✅ Training monitoring dashboard
- ✅ Unit tests for data loaders and models
- ✅ Documentation (architecture.md, setup instructions)

**Validation Criteria:**
- Training loss < 0.1
- Validation loss < 0.15
- Model can reconstruct normal samples with MSE < 0.05
- All tests passing (pytest)
- Dashboard shows live training metrics

---

## 📅 WEEK 2: Anomaly Detection & Root Cause Analysis

### Phase 2A: Anomaly Detection System (Days 6-8)

#### Person 1 (ML Engineer) - LEAD

**Priority Tasks:**
1. ✅ Implement anomaly scoring mechanism
2. ✅ Develop threshold calibration (percentile-based, MAD, EVT)
3. ✅ Create evaluation metrics (ROC-AUC, PR-AUC, F1)
4. ✅ Implement advanced threshold methods
5. ✅ Fine-tune model based on Week 1 results

**Files to Create:**
- `src/inference/anomaly_detector.py`
- `src/inference/threshold_calibrator.py`
- `notebooks/03_model_evaluation.ipynb`

**Code Reference:**
```python
# src/inference/anomaly_detector.py
import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pickle

class AnomalyDetector:
    """Anomaly detection using trained VAE"""

    def __init__(self, model_path: str, threshold_percentile: float = 95.0):
        self.model = self._load_model(model_path)
        self.model.eval()
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.normal_scores_stats = None

    def _load_model(self, path: str):
        from src.models.vae import TimeSeriesVAE
        model = TimeSeriesVAE.load_from_checkpoint(path)
        return model

    def calibrate_threshold(self, normal_data_loader):
        """Calibrate anomaly threshold on normal data"""
        scores = []

        with torch.no_grad():
            for batch, _ in normal_data_loader:
                batch_scores = self._compute_batch_scores(batch)
                scores.extend(batch_scores.cpu().numpy())

        scores = np.array(scores)
        self.threshold = np.percentile(scores, self.threshold_percentile)
        self.normal_scores_stats = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'median': np.median(scores),
            'percentile_95': np.percentile(scores, 95),
            'percentile_99': np.percentile(scores, 99)
        }

        return self.threshold

    def _compute_batch_scores(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction, mu, logvar = self.model(x)
        scores = self.model.compute_anomaly_score(x, reconstruction, mu, logvar)
        return scores

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies. Returns: (anomaly_scores, is_anomaly)"""
        with torch.no_grad():
            scores = self._compute_batch_scores(x).cpu().numpy()

        is_anomaly = scores > self.threshold
        return scores, is_anomaly

    def evaluate(self, test_loader) -> Dict[str, float]:
        """Evaluate on test set with labels"""
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch, labels in test_loader:
                scores = self._compute_batch_scores(batch)
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # ROC-AUC
        roc_auc = roc_auc_score(all_labels, all_scores)

        # PR-AUC
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        pr_auc = auc(recall, precision)

        # F1 at current threshold
        predictions = (all_scores > self.threshold).astype(int)
        tp = np.sum((predictions == 1) & (all_labels == 1))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        fn = np.sum((predictions == 0) & (all_labels == 1))

        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision_score * recall_score / (precision_score + recall_score) \
                   if (precision_score + recall_score) > 0 else 0

        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'threshold': self.threshold
        }

    def save_calibration(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'threshold': self.threshold,
                'threshold_percentile': self.threshold_percentile,
                'normal_scores_stats': self.normal_scores_stats
            }, f)

    def load_calibration(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.threshold = data['threshold']
        self.threshold_percentile = data['threshold_percentile']
        self.normal_scores_stats = data['normal_scores_stats']
```

```python
# src/inference/threshold_calibrator.py
import numpy as np
from scipy import stats

class AdaptiveThresholdCalibrator:
    """Advanced threshold calibration strategies"""

    @staticmethod
    def percentile_threshold(scores: np.ndarray, percentile: float = 95) -> float:
        return np.percentile(scores, percentile)

    @staticmethod
    def mad_threshold(scores: np.ndarray, n_sigma: float = 3.0) -> float:
        """Median Absolute Deviation (robust to outliers)"""
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        return median + n_sigma * mad / 0.6745

    @staticmethod
    def gaussian_threshold(scores: np.ndarray, n_sigma: float = 3.0) -> float:
        return np.mean(scores) + n_sigma * np.std(scores)

    @staticmethod
    def extreme_value_threshold(scores: np.ndarray, quantile: float = 0.95) -> float:
        """Extreme Value Theory based"""
        params = stats.genextreme.fit(scores)
        return stats.genextreme.ppf(quantile, *params)

    @staticmethod
    def dynamic_threshold(scores: np.ndarray, window_size: int = 100,
                         n_sigma: float = 3.0) -> np.ndarray:
        """Dynamic threshold for streaming"""
        thresholds = []
        for i in range(len(scores)):
            start = max(0, i - window_size)
            window = scores[start:i+1]
            if len(window) < 10:
                thresholds.append(np.inf)
            else:
                thresholds.append(np.mean(window) + n_sigma * np.std(window))
        return np.array(thresholds)
```

---

#### Person 2 (Data Engineer) - SUPPORT

**Priority Tasks:**
1. ✅ Create comprehensive evaluation scripts
2. ✅ Implement cross-validation pipeline
3. ✅ Set up model registry with DVC
4. ✅ Create batch inference pipeline

**Files to Create:**
- `scripts/evaluate.py`
- `scripts/cross_validate.py`
- `.dvc/config` (model registry)

**Code Reference:**
```python
# scripts/evaluate.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse

from src.inference.anomaly_detector import AnomalyDetector
from src.data.loaders import TurbofanDataset

def evaluate_model(model_path: str, test_data_path: str, output_dir: str):
    """Comprehensive model evaluation"""

    # Load test data
    test_dataset = TurbofanDataset(test_data_path, sequence_length=50)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize detector
    detector = AnomalyDetector(model_path, threshold_percentile=95)

    # Calibrate (use first 70% as normal)
    normal_size = int(len(test_dataset) * 0.7)
    normal_dataset = torch.utils.data.Subset(test_dataset, range(normal_size))
    normal_loader = DataLoader(normal_dataset, batch_size=128)

    print("Calibrating threshold...")
    detector.calibrate_threshold(normal_loader)

    # Evaluate
    print("Evaluating model...")
    metrics = detector.evaluate(test_loader)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*50}")
    print("Evaluation Results:")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}")
    print(f"{'='*50}")

    # Save calibration
    detector.save_calibration(output_path / 'calibration.pkl')

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to model checkpoint')
    parser.add_argument('--test_data', required=True, help='Path to test data')
    parser.add_argument('--output_dir', default='evaluation_results', help='Output directory')

    args = parser.parse_args()
    evaluate_model(args.model_path, args.test_data, args.output_dir)
```

**Run Evaluation:**
```bash
python scripts/evaluate.py \
    --model_path models/checkpoints/best_model.ckpt \
    --test_data data/raw/nasa_turbofan/test_FD001.txt \
    --output_dir evaluation_results
```

---

### Phase 2B: Root Cause Analysis (Days 9-10)

#### Person 1 (ML Engineer) - LEAD

**Priority Tasks:**
1. ✅ Implement per-sensor reconstruction error attribution
2. ✅ Develop latent sensitivity analysis
3. ✅ Integrate SHAP for explainability
4. ✅ Create temporal contribution analysis

**Files to Create:**
- `src/inference/root_cause_analyzer.py`
- `notebooks/04_root_cause_analysis.ipynb`

**Code Reference:**
```python
# src/inference/root_cause_analyzer.py
import torch
import numpy as np
from typing import Dict, List
import shap

class RootCauseAnalyzer:
    """Root cause analysis for detected anomalies"""

    def __init__(self, model, sensor_names: List[str]):
        self.model = model
        self.model.eval()
        self.sensor_names = sensor_names

    def analyze_reconstruction_error(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Compute per-sensor reconstruction error"""
        with torch.no_grad():
            reconstruction, mu, logvar = self.model(x)

        # Per-sensor error [batch, sensors, time]
        error = (x - reconstruction) ** 2

        # Aggregate over time
        sensor_errors = error.mean(dim=2).cpu().numpy()  # [batch, sensors]
        avg_sensor_errors = sensor_errors.mean(axis=0)  # [sensors]

        return {
            'sensor_errors': sensor_errors,
            'avg_sensor_errors': avg_sensor_errors,
            'sensor_ranking': np.argsort(avg_sensor_errors)[::-1],
            'top_sensors': [self.sensor_names[i] for i in np.argsort(avg_sensor_errors)[::-1][:5]]
        }

    def analyze_temporal_contribution(self, x: torch.Tensor,
                                     window_size: int = 10) -> Dict[str, np.ndarray]:
        """Analyze which time windows contribute most"""
        with torch.no_grad():
            reconstruction, mu, logvar = self.model(x)

        error = (x - reconstruction) ** 2
        seq_length = error.shape[2]
        num_windows = seq_length - window_size + 1

        window_errors = []
        for i in range(num_windows):
            window_error = error[:, :, i:i+window_size].mean(dim=(1, 2))
            window_errors.append(window_error)

        window_errors = torch.stack(window_errors, dim=1).cpu().numpy()

        return {
            'window_errors': window_errors,
            'peak_window_idx': np.argmax(window_errors.mean(axis=0)),
            'window_size': window_size
        }

    def analyze_latent_sensitivity(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Analyze sensitivity to latent dimensions"""
        with torch.no_grad():
            mu, logvar = self.model.encoder(x)
            z = self.model.reparameterize(mu, logvar)

        z.requires_grad_(True)
        reconstruction = self.model.decoder(z)

        sensitivities = []
        for i in range(reconstruction.shape[0]):
            jacobian = torch.autograd.functional.jacobian(
                lambda z_: self.model.decoder(z_)[i],
                z[i:i+1]
            )
            sensitivity = torch.norm(jacobian, dim=(1, 2, 3)).detach()
            sensitivities.append(sensitivity)

        sensitivities = torch.stack(sensitivities).cpu().numpy()

        return {
            'latent_sensitivities': sensitivities,
            'most_sensitive_dims': np.argsort(sensitivities.mean(axis=0))[::-1][:10]
        }

    def generate_report(self, x: torch.Tensor, anomaly_idx: int) -> Dict:
        """Generate comprehensive root cause report"""
        sample = x[anomaly_idx:anomaly_idx+1]

        recon_analysis = self.analyze_reconstruction_error(sample)
        temporal_analysis = self.analyze_temporal_contribution(sample)
        latent_analysis = self.analyze_latent_sensitivity(sample)

        return {
            'anomaly_index': anomaly_idx,
            'top_contributing_sensors': recon_analysis['top_sensors'],
            'sensor_error_scores': {
                self.sensor_names[i]: float(recon_analysis['avg_sensor_errors'][i])
                for i in range(len(self.sensor_names))
            },
            'peak_time_window': int(temporal_analysis['peak_window_idx']),
            'most_sensitive_latent_dims': latent_analysis['most_sensitive_dims'].tolist()
        }
```

---

#### Person 3 (Full-Stack) - LEAD

**Priority Tasks:**
1. ✅ Build anomaly detection dashboard
2. ✅ Create root cause visualization interface
3. ✅ Implement interactive anomaly explorer
4. ✅ Add CSV upload functionality

**Files to Create:**
- `dashboards/pages/2_anomaly_detection.py`
- `dashboards/pages/3_root_cause.py`
- `dashboards/components/plots.py`

**Code Reference:**
```python
# dashboards/pages/2_anomaly_detection.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import torch
import numpy as np
import pandas as pd
from src.inference.anomaly_detector import AnomalyDetector

st.title("🔍 Anomaly Detection Dashboard")

# Load model
@st.cache_resource
def load_detector():
    model_path = st.sidebar.text_input("Model Path", "models/best_model.ckpt")
    if model_path:
        detector = AnomalyDetector(model_path)
        detector.load_calibration('models/calibration.pkl')
        return detector
    return None

detector = load_detector()

if detector:
    st.success("✅ Model loaded successfully!")

    # Upload data
    uploaded_file = st.file_uploader("Upload sensor data (CSV)", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Preview")
        st.dataframe(df.head())

        if st.button("🔍 Detect Anomalies"):
            with st.spinner("Analyzing..."):
                # Assume df has correct format: [samples x sensors]
                # Reshape to sequences if needed
                # x = ... (preprocessing)

                # For demo: create dummy tensor
                x = torch.FloatTensor(df.values[:100]).unsqueeze(0)

                scores, is_anomaly = detector.predict(x)

                # Visualization
                col1, col2 = st.columns(2)

                with col1:
                    fig = go.Figure()
                    colors = ['red' if a else 'green' for a in is_anomaly]
                    fig.add_trace(go.Scatter(
                        y=scores,
                        mode='lines+markers',
                        name='Anomaly Score',
                        marker=dict(color=colors, size=8),
                        line=dict(color='blue')
                    ))
                    fig.add_hline(y=detector.threshold, line_dash="dash",
                                 annotation_text="Threshold", line_color="red")
                    fig.update_layout(
                        title="Anomaly Scores Over Time",
                        xaxis_title="Sample Index",
                        yaxis_title="Anomaly Score"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.metric("Total Anomalies", int(is_anomaly.sum()),
                             delta=f"{is_anomaly.mean()*100:.1f}% of samples")
                    st.metric("Max Score", f"{scores.max():.4f}")
                    st.metric("Threshold", f"{detector.threshold:.4f}")
                    st.metric("Mean Score", f"{scores.mean():.4f}")

                # Anomaly list
                st.subheader("Detected Anomalies")
                anomaly_indices = np.where(is_anomaly)[0]

                if len(anomaly_indices) > 0:
                    anomaly_df = pd.DataFrame({
                        'Index': anomaly_indices,
                        'Score': scores[anomaly_indices],
                        'Severity': ['High' if s > detector.threshold * 1.5 else 'Medium'
                                    for s in scores[anomaly_indices]]
                    })
                    st.dataframe(anomaly_df, use_container_width=True)
                else:
                    st.success("✅ No anomalies detected!")
else:
    st.warning("Please configure model path in sidebar")
```

```python
# dashboards/pages/3_root_cause.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import torch
from src.inference.root_cause_analyzer import RootCauseAnalyzer
from src.models.vae import TimeSeriesVAE

st.title("🎯 Root Cause Analysis")

# Load model and analyzer
@st.cache_resource
def load_analyzer():
    model = TimeSeriesVAE.load_from_checkpoint('models/best_model.ckpt')
    sensor_names = [f'Sensor_{i}' for i in range(1, 15)]
    return RootCauseAnalyzer(model, sensor_names)

analyzer = load_analyzer()

st.sidebar.header("Settings")
anomaly_idx = st.sidebar.number_input("Anomaly Index", min_value=0, value=0, step=1)

# For demo: load sample data
# x = ... (load your data)

if st.sidebar.button("🔬 Analyze Root Cause"):
    with st.spinner("Analyzing root cause..."):
        # Demo tensor
        x = torch.randn(10, 14, 50)

        report = analyzer.generate_report(x, anomaly_idx)

        st.subheader("Analysis Report")

        # Top contributing sensors
        col1, col2 = st.columns([2, 1])

        with col1:
            sensor_scores = report['sensor_error_scores']
            fig = px.bar(
                x=list(sensor_scores.keys()),
                y=list(sensor_scores.values()),
                labels={'x': 'Sensor', 'y': 'Reconstruction Error'},
                title='Sensor-wise Contribution to Anomaly',
                color=list(sensor_scores.values()),
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Top 5 Sensors")
            for i, sensor in enumerate(report['top_contributing_sensors'], 1):
                score = sensor_scores[sensor]
                st.metric(f"{i}. {sensor}", f"{score:.4f}")

        # Temporal analysis
        st.subheader("Temporal Analysis")
        st.info(f"🕐 Peak anomaly detected at time window: **{report['peak_time_window']}**")

        # Latent space
        st.subheader("Latent Space Analysis")
        st.write("Most sensitive latent dimensions:")
        st.write(report['most_sensitive_latent_dims'])

        # Download report
        import json
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="📥 Download Report (JSON)",
            data=report_json,
            file_name=f"root_cause_report_{anomaly_idx}.json",
            mime="application/json"
        )
```

---

### Week 2 Deliverables

**Completed by End of Week 2:**
- ✅ Anomaly detection system with multiple threshold calibration methods
- ✅ Root cause analysis with per-sensor attribution
- ✅ Evaluation metrics on test set (ROC-AUC > 0.85 target)
- ✅ Interactive anomaly detection dashboard
- ✅ Root cause analysis dashboard
- ✅ Comprehensive evaluation reports
- ✅ Integration tests for inference pipeline

**Validation Criteria:**
- ROC-AUC > 0.85 on test set
- Precision > 0.8, Recall > 0.75
- Root cause analysis identifies correct sensor 80%+ of time
- Dashboard allows upload and inference on custom data
- All tests passing

---

## 📅 WEEK 3: Vision VAE & Drift Monitoring

### Phase 3A: Vision-Based Anomaly Detection (Days 11-13)

#### Person 1 (ML Engineer) - LEAD

**Priority Tasks:**
1. ✅ Implement 2D-CNN VAE for images
2. ✅ Adapt architecture for MVTec AD dataset
3. ✅ Implement SSIM loss for better reconstruction
4. ✅ Train vision VAE
5. ✅ Evaluate on multiple MVTec categories

**Files to Create:**
- `src/models/vision_vae.py`
- `src/data/vision_loaders.py`
- `configs/model/vae_vision.yaml`
- `scripts/train_vision.py`

**Code Reference:**
```python
# src/models/vision_vae.py
import torch
import torch.nn as nn
import pytorch_lightning as pl

class VisionEncoder(nn.Module):
    """2D-CNN Encoder for images"""

    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            # [B, 3, 256, 256] -> [B, 64, 128, 128]
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # [B, 64, 128, 128] -> [B, 128, 64, 64]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # [B, 128, 64, 64] -> [B, 256, 32, 32]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # [B, 256, 32, 32] -> [B, 512, 16, 16]
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # [B, 512, 16, 16] -> [B, 512, 8, 8]
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.flatten_size = 512 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.flatten_size)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VisionDecoder(nn.Module):
    """2D-CNN Decoder for images"""

    def __init__(self, latent_dim: int = 256, out_channels: int = 3):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 512, 8, 8)
        return self.decoder(h)


class VisionVAE(pl.LightningModule):
    """VAE for vision-based anomaly detection"""

    def __init__(self, in_channels: int = 3, latent_dim: int = 256,
                 beta: float = 1.0, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = VisionEncoder(in_channels, latent_dim)
        self.decoder = VisionDecoder(latent_dim, in_channels)

        self.beta = beta
        self.learning_rate = learning_rate

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def compute_loss(self, x, reconstruction, mu, logvar):
        # MSE reconstruction
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='mean')

        # SSIM loss
        from pytorch_msssim import ssim
        ssim_loss = 1 - ssim(reconstruction, x, data_range=1.0, size_average=True)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + 0.1 * ssim_loss + self.beta * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'ssim_loss': ssim_loss,
            'kl_loss': kl_loss
        }

    def training_step(self, batch, batch_idx):
        x, _ = batch
        reconstruction, mu, logvar = self(x)
        losses = self.compute_loss(x, reconstruction, mu, logvar)

        self.log_dict({f'train/{k}': v for k, v in losses.items()}, prog_bar=True)
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        reconstruction, mu, logvar = self(x)
        losses = self.compute_loss(x, reconstruction, mu, logvar)

        self.log_dict({f'val/{k}': v for k, v in losses.items()}, prog_bar=True)

        # Log sample reconstructions
        if batch_idx == 0:
            self._log_images(x, reconstruction)

        return losses

    def _log_images(self, x, reconstruction):
        import wandb
        images = torch.cat([x[:4], reconstruction[:4]])
        grid = torchvision.utils.make_grid(images, nrow=4)
        self.logger.experiment.log({"reconstructions": [wandb.Image(grid)]})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
```

```python
# src/data/vision_loaders.py
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

class MVTecDataset(Dataset):
    """MVTec Anomaly Detection Dataset"""

    def __init__(self, root_dir: str, category: str = 'bottle',
                 split: str = 'train', image_size: int = 256):
        self.root_dir = Path(root_dir) / category / split
        self.image_size = image_size

        self.image_paths = list(self.root_dir.glob('**/*.png'))
        self.labels = [0 if 'good' in str(p) else 1 for p in self.image_paths]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
```

---

#### Person 2 (Data Engineer) - SUPPORT

**Priority Tasks:**
1. ✅ Download and prepare MVTec AD dataset
2. ✅ Implement vision-specific data augmentations
3. ✅ Create image preprocessing pipeline
4. ✅ Set up vision model training infrastructure

**Files to Create:**
- `src/data/vision_augmentations.py`
- `src/data/vision_preprocessors.py`
- Update `scripts/download_datasets.sh`

**Code Reference:**
```python
# src/data/vision_augmentations.py
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

class VisionAugmentation:
    """Data augmentation for industrial images"""

    def __init__(self, augment_prob: float = 0.5):
        self.augment_prob = augment_prob

    def __call__(self, img):
        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)

        # Random rotation
        if random.random() > self.augment_prob:
            angle = random.uniform(-10, 10)
            img = TF.rotate(img, angle)

        # Color jitter
        if random.random() > self.augment_prob:
            img = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )(img)

        # Gaussian blur
        if random.random() > self.augment_prob:
            img = TF.gaussian_blur(img, kernel_size=5)

        return img
```

---

### Phase 3B: Drift Monitoring System (Days 14-15)

#### Person 2 (Data Engineer) - LEAD

**Priority Tasks:**
1. ✅ Implement latent distribution drift detection (MMD, KS test, Wasserstein)
2. ✅ Create reconstruction error drift monitoring
3. ✅ Develop retraining triggers and automation
4. ✅ Implement production metrics tracking

**Files to Create:**
- `src/monitoring/drift_detector.py`
- `src/monitoring/metrics.py`
- `scripts/monitor_drift.py`

**Code Reference:**
```python
# src/monitoring/drift_detector.py
import numpy as np
from scipy import stats
from typing import Dict, Tuple

class DriftDetector:
    """Detect distribution drift in production"""

    def __init__(self, reference_window_size: int = 1000):
        self.reference_window_size = reference_window_size
        self.reference_latents = None
        self.reference_scores = None

    def set_reference(self, latents: np.ndarray, scores: np.ndarray):
        """Set reference distribution from training/validation"""
        self.reference_latents = latents
        self.reference_scores = scores

    def detect_latent_drift(self, current_latents: np.ndarray,
                           method: str = 'mmd') -> Dict[str, float]:
        """Detect drift in latent space"""

        if method == 'mmd':
            drift_score = self._maximum_mean_discrepancy(
                self.reference_latents, current_latents
            )
        elif method == 'ks':
            drift_score = self._kolmogorov_smirnov(
                self.reference_latents, current_latents
            )
        elif method == 'wasserstein':
            drift_score = self._wasserstein_distance(
                self.reference_latents, current_latents
            )

        return {'drift_score': drift_score, 'method': method}

    @staticmethod
    def _maximum_mean_discrepancy(X: np.ndarray, Y: np.ndarray,
                                  gamma: float = 1.0) -> float:
        """Compute MMD between two distributions"""
        XX = np.dot(X, X.T)
        YY = np.dot(Y, Y.T)
        XY = np.dot(X, Y.T)

        K_XX = np.exp(-gamma * (XX.diagonal().reshape(-1, 1) +
                                XX.diagonal().reshape(1, -1) - 2 * XX))
        K_YY = np.exp(-gamma * (YY.diagonal().reshape(-1, 1) +
                                YY.diagonal().reshape(1, -1) - 2 * YY))
        K_XY = np.exp(-gamma * (XX.diagonal().reshape(-1, 1) +
                                YY.diagonal().reshape(1, -1) - 2 * XY))

        mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        return float(mmd)

    @staticmethod
    def _kolmogorov_smirnov(X: np.ndarray, Y: np.ndarray) -> float:
        """KS test for each dimension, return max"""
        ks_stats = []
        for dim in range(X.shape[1]):
            stat, _ = stats.ks_2samp(X[:, dim], Y[:, dim])
            ks_stats.append(stat)
        return float(np.max(ks_stats))

    @staticmethod
    def _wasserstein_distance(X: np.ndarray, Y: np.ndarray) -> float:
        """Wasserstein distance averaged over dimensions"""
        distances = []
        for dim in range(X.shape[1]):
            dist = stats.wasserstein_distance(X[:, dim], Y[:, dim])
            distances.append(dist)
        return float(np.mean(distances))

    def detect_score_drift(self, current_scores: np.ndarray) -> Dict[str, float]:
        """Detect drift in anomaly scores"""
        ks_stat, ks_pval = stats.ks_2samp(self.reference_scores, current_scores)

        ref_mean = np.mean(self.reference_scores)
        cur_mean = np.mean(current_scores)
        mean_shift = (cur_mean - ref_mean) / (np.std(self.reference_scores) + 1e-8)

        ref_var = np.var(self.reference_scores)
        cur_var = np.var(current_scores)
        var_ratio = cur_var / (ref_var + 1e-8)

        return {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'mean_shift': mean_shift,
            'variance_ratio': var_ratio,
            'drift_detected': ks_pval < 0.05 or abs(mean_shift) > 2
        }

    def should_retrain(self, drift_results: Dict) -> Tuple[bool, str]:
        """Determine if model should be retrained"""
        reasons = []

        if 'drift_score' in drift_results and drift_results['drift_score'] > 0.1:
            reasons.append(f"Latent drift: {drift_results['drift_score']:.4f}")

        if drift_results.get('drift_detected', False):
            reasons.append("Score distribution drift")

        if abs(drift_results.get('mean_shift', 0)) > 3:
            reasons.append(f"Mean shift: {drift_results['mean_shift']:.2f}σ")

        should_retrain = len(reasons) > 0
        reason_str = "; ".join(reasons) if reasons else "No drift"

        return should_retrain, reason_str
```

```python
# src/monitoring/metrics.py
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ProductionMetrics:
    timestamp: datetime
    num_predictions: int
    num_anomalies: int
    avg_score: float
    max_score: float
    latency_ms: float
    drift_score: float

class MetricsTracker:
    """Track production metrics"""

    def __init__(self):
        self.metrics_history: List[ProductionMetrics] = []

    def log_batch(self, scores: np.ndarray, latency_ms: float,
                  drift_score: float = 0.0, threshold: float = 0.5):
        metrics = ProductionMetrics(
            timestamp=datetime.now(),
            num_predictions=len(scores),
            num_anomalies=int((scores > threshold).sum()),
            avg_score=float(scores.mean()),
            max_score=float(scores.max()),
            latency_ms=latency_ms,
            drift_score=drift_score
        )
        self.metrics_history.append(metrics)

    def get_summary(self, window_hours: int = 24) -> Dict:
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent = [m for m in self.metrics_history if m.timestamp > cutoff]

        if not recent:
            return {}

        return {
            'total_predictions': sum(m.num_predictions for m in recent),
            'total_anomalies': sum(m.num_anomalies for m in recent),
            'avg_latency_ms': np.mean([m.latency_ms for m in recent]),
            'max_drift_score': max(m.drift_score for m in recent),
            'anomaly_rate': sum(m.num_anomalies for m in recent) /
                          sum(m.num_predictions for m in recent)
        }
```

---

#### Person 3 (Full-Stack) - SUPPORT

**Priority Tasks:**
1. ✅ Create drift monitoring dashboard
2. ✅ Build vision VAE demo interface
3. ✅ Implement alerting system for drift
4. ✅ Add Prometheus metrics export

**Files to Create:**
- `dashboards/pages/4_drift_monitoring.py`
- `dashboards/pages/5_vision_demo.py`

**Code Reference:**
```python
# dashboards/pages/4_drift_monitoring.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

st.title("📈 Drift Monitoring Dashboard")

# Simulated data (replace with real)
@st.cache_data(ttl=60)
def get_drift_metrics():
    timestamps = pd.date_range(end=datetime.now(), periods=100, freq='1H')
    return {
        'timestamps': timestamps,
        'drift_scores': np.random.exponential(0.05, 100),
        'anomaly_rates': np.random.beta(2, 20, 100),
        'mean_scores': np.random.normal(0.3, 0.05, 100)
    }

metrics = get_drift_metrics()

# Drift score over time
st.subheader("Latent Distribution Drift (MMD)")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=metrics['timestamps'],
    y=metrics['drift_scores'],
    mode='lines',
    name='MMD Score',
    fill='tozeroy'
))
fig.add_hline(y=0.1, line_dash="dash", annotation_text="⚠️ Threshold",
             line_color="red")
fig.update_layout(xaxis_title="Time", yaxis_title="Drift Score")
st.plotly_chart(fig, use_container_width=True)

# Anomaly rate trend
st.subheader("Anomaly Rate Trend")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=metrics['timestamps'],
    y=metrics['anomaly_rates'] * 100,
    mode='lines',
    fill='tozeroy',
    name='Anomaly Rate %'
))
fig.update_layout(xaxis_title="Time", yaxis_title="Anomaly Rate (%)")
st.plotly_chart(fig, use_container_width=True)

# Alerts
st.subheader("⚠️ System Alerts")
current_drift = metrics['drift_scores'][-1]
if current_drift > 0.1:
    st.error(f"🚨 HIGH DRIFT DETECTED: {current_drift:.4f} > 0.1 threshold")
    st.warning("**Recommendation:** Consider retraining the model")
else:
    st.success("✅ No significant drift detected")

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Drift", f"{current_drift:.4f}",
             delta=f"{(current_drift - metrics['drift_scores'][-10])*100:.1f}%")
with col2:
    st.metric("Anomaly Rate (24h)", f"{metrics['anomaly_rates'].mean()*100:.2f}%")
with col3:
    days_since_training = 7
    st.metric("Days Since Training", days_since_training)
with col4:
    if current_drift > 0.1:
        st.metric("Status", "⚠️ Action Required")
    else:
        st.metric("Status", "✅ Healthy")

# Retraining
if st.button("🔄 Trigger Model Retraining"):
    with st.spinner("Submitting retraining job..."):
        import time
        time.sleep(2)
        st.success("✅ Retraining job submitted successfully!")
```

---

### Week 3 Deliverables

**Completed by End of Week 3:**
- ✅ Vision VAE trained on MVTec AD dataset
- ✅ Drift detection system (MMD, KS, Wasserstein)
- ✅ Production metrics tracking
- ✅ Drift monitoring dashboard with alerts
- ✅ Vision anomaly detection demo
- ✅ Retraining automation logic
- ✅ Integration tests for monitoring

**Validation Criteria:**
- Vision VAE achieves >0.90 ROC-AUC on MVTec
- Drift detector correctly identifies simulated drift
- Dashboard updates in real-time
- Alerts trigger at correct thresholds
- All tests passing

---

## 📅 WEEK 4: Multimodal Fusion & Production Deployment

### Phase 4A: Multimodal VAE (Days 16-18)

#### Person 1 (ML Engineer) - LEAD

**Priority Tasks:**
1. ✅ Implement multimodal VAE with Product-of-Experts fusion
2. ✅ Create cross-modal attention mechanism
3. ✅ Implement alternative fusion strategies (MoE, concatenation)
4. ✅ Train multimodal VAE on combined dataset
5. ✅ Evaluate fusion vs single-modality performance

**Files to Create:**
- `src/models/multimodal_vae.py`
- `src/data/multimodal_loader.py`
- `configs/model/vae_multimodal.yaml`
- `scripts/train_multimodal.py`

**Code Reference:**
```python
# src/models/multimodal_vae.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Tuple, Optional

class MultimodalVAE(pl.LightningModule):
    """Multimodal VAE with Product-of-Experts fusion"""

    def __init__(self,
                 timeseries_encoder,
                 vision_encoder,
                 shared_latent_dim: int = 256,
                 fusion_method: str = 'poe',
                 beta: float = 1.0,
                 learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['timeseries_encoder', 'vision_encoder'])

        self.ts_encoder = timeseries_encoder
        self.vis_encoder = vision_encoder

        self.shared_latent_dim = shared_latent_dim
        self.fusion_method = fusion_method

        # Cross-modal attention
        self.cross_attention = CrossModalAttention(shared_latent_dim)

        # Decoders
        from .decoders import Conv1DDecoder
        from .vision_vae import VisionDecoder

        self.ts_decoder = Conv1DDecoder(shared_latent_dim, 14, 50)
        self.vis_decoder = VisionDecoder(shared_latent_dim, 3)

        self.beta = beta
        self.learning_rate = learning_rate

    def encode(self, ts_data: Optional[torch.Tensor] = None,
               vis_data: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode multimodal data into shared latent"""
        mus, logvars = [], []

        if ts_data is not None:
            ts_mu, ts_logvar = self.ts_encoder(ts_data)
            mus.append(ts_mu)
            logvars.append(ts_logvar)

        if vis_data is not None:
            vis_mu, vis_logvar = self.vis_encoder(vis_data)
            mus.append(vis_mu)
            logvars.append(vis_logvar)

        # Fusion
        if self.fusion_method == 'poe':
            mu, logvar = self._product_of_experts(mus, logvars)
        elif self.fusion_method == 'moe':
            mu, logvar = self._mixture_of_experts(mus, logvars)
        elif self.fusion_method == 'concat':
            mu = torch.cat(mus, dim=-1)
            logvar = torch.cat(logvars, dim=-1)

        return mu, logvar

    def _product_of_experts(self, mus: list, logvars: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Product of Experts fusion (precision-weighted)"""
        precisions = [torch.exp(-lv) for lv in logvars]
        precision_sum = sum(precisions)

        mu = sum([p * m for p, m in zip(precisions, mus)]) / (precision_sum + 1e-8)
        var = 1.0 / (precision_sum + 1e-8)
        logvar = torch.log(var)

        return mu, logvar

    def _mixture_of_experts(self, mus: list, logvars: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixture of Experts (simple averaging)"""
        mu = torch.stack(mus).mean(dim=0)
        logvar = torch.stack(logvars).mean(dim=0)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode latent into both modalities"""
        return {
            'timeseries': self.ts_decoder(z),
            'vision': self.vis_decoder(z)
        }

    def forward(self, ts_data: Optional[torch.Tensor] = None,
                vis_data: Optional[torch.Tensor] = None):
        mu, logvar = self.encode(ts_data, vis_data)
        z = self.reparameterize(mu, logvar)
        reconstructions = self.decode(z)
        return reconstructions, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_loss(self, ts_data, vis_data, reconstructions, mu, logvar):
        losses = {}

        if ts_data is not None:
            ts_recon_loss = nn.functional.mse_loss(reconstructions['timeseries'], ts_data)
            losses['ts_recon'] = ts_recon_loss

        if vis_data is not None:
            vis_recon_loss = nn.functional.mse_loss(reconstructions['vision'], vis_data)
            losses['vis_recon'] = vis_recon_loss

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        losses['kl'] = kl_loss

        total = sum(losses.values()) + self.beta * kl_loss
        losses['loss'] = total

        return losses

    def training_step(self, batch, batch_idx):
        ts_data, vis_data, _ = batch
        reconstructions, mu, logvar = self(ts_data, vis_data)
        losses = self.compute_loss(ts_data, vis_data, reconstructions, mu, logvar)

        self.log_dict({f'train/{k}': v for k, v in losses.items()}, prog_bar=True)
        return losses['loss']

    def compute_anomaly_score(self, ts_data, vis_data, reconstructions, mu, logvar):
        """Compute multimodal anomaly score"""
        scores = []

        if ts_data is not None:
            ts_error = torch.mean((ts_data - reconstructions['timeseries']) ** 2, dim=(1, 2))
            scores.append(ts_error)

        if vis_data is not None:
            vis_error = torch.mean((vis_data - reconstructions['vision']) ** 2, dim=(1, 2, 3))
            scores.append(vis_error)

        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        scores.append(kl_div)

        return sum(scores) / len(scores)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


class CrossModalAttention(nn.Module):
    """Cross-modal attention"""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        attn_out, _ = self.attention(query, key, value)
        attn_out = attn_out.squeeze(1)

        return self.norm(query.squeeze(1) + attn_out)
```

```python
# src/data/multimodal_loader.py
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    """Combined time-series + vision dataset"""

    def __init__(self, ts_dataset, vision_dataset):
        self.ts_dataset = ts_dataset
        self.vision_dataset = vision_dataset
        assert len(ts_dataset) == len(vision_dataset)

    def __len__(self):
        return len(self.ts_dataset)

    def __getitem__(self, idx):
        ts_data, ts_label = self.ts_dataset[idx]
        vis_data, vis_label = self.vision_dataset[idx]
        label = max(ts_label, vis_label)  # Anomaly if either is anomalous
        return ts_data, vis_data, label
```

---

### Phase 4B: Production Deployment (Days 19-20)

#### Person 3 (Full-Stack) - LEAD

**Priority Tasks:**
1. ✅ Create FastAPI inference service
2. ✅ Implement ONNX export and optimization
3. ✅ Build production Docker containers
4. ✅ Set up Nginx load balancing
5. ✅ Configure Prometheus + Grafana monitoring
6. ✅ Write deployment documentation

**Files to Create:**
- `src/api/main.py`
- `src/api/routes.py`
- `src/api/schemas.py`
- `docker/Dockerfile.serve`
- `docker/Dockerfile.dashboard`
- `docker/docker-compose.yml`
- `docker/nginx.conf`
- `docker/prometheus.yml`
- `scripts/export_onnx.py`
- `docs/deployment_guide.md`

**Code Reference:**
```python
# src/api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import numpy as np
from typing import List
import io
from PIL import Image

from src.inference.anomaly_detector import AnomalyDetector
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.metrics import MetricsTracker
from .schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="VAE Anomaly Detection API", version="1.0.0")

detector = None
drift_detector = None
metrics_tracker = MetricsTracker()

@app.on_event("startup")
async def load_models():
    global detector, drift_detector
    detector = AnomalyDetector('models/best_model.ckpt')
    detector.load_calibration('models/calibration.pkl')
    drift_detector = DriftDetector()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": detector is not None}

@app.post("/predict/timeseries", response_model=PredictionResponse)
async def predict_timeseries(request: PredictionRequest):
    try:
        data = torch.FloatTensor(request.data).unsqueeze(0)

        import time
        start = time.time()
        scores, is_anomaly = detector.predict(data)
        latency_ms = (time.time() - start) * 1000

        metrics_tracker.log_batch(scores, latency_ms, threshold=detector.threshold)

        return PredictionResponse(
            anomaly_scores=scores.tolist(),
            is_anomaly=is_anomaly.tolist(),
            threshold=detector.threshold,
            latency_ms=latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/vision")
async def predict_vision(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)

        # Vision prediction logic here...

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    return metrics_tracker.get_summary(window_hours=24)

@app.post("/drift/check")
async def check_drift(current_data: List[List[float]]):
    current_array = np.array(current_data)
    drift_results = drift_detector.detect_score_drift(current_array.flatten())
    should_retrain, reason = drift_detector.should_retrain(drift_results)

    return {
        "drift_detected": drift_results.get('drift_detected', False),
        "should_retrain": should_retrain,
        "reason": reason
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```python
# src/api/schemas.py
from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    data: List[List[float]]

class PredictionResponse(BaseModel):
    anomaly_scores: List[float]
    is_anomaly: List[bool]
    threshold: float
    latency_ms: float
```

```dockerfile
# docker/Dockerfile.serve
FROM python:3.10-slim

WORKDIR /app

COPY requirements/prod.txt .
RUN pip install --no-cache-dir -r prod.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.serve
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/best_model.ckpt
    volumes:
      - ../models:/app/models:ro
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G

  dashboard:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dashboard
    ports:
      - "8501:8501"
    depends_on:
      - api

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
```

```python
# scripts/export_onnx.py
import torch
from src.models.vae import TimeSeriesVAE

def export_to_onnx(checkpoint_path: str, output_path: str):
    model = TimeSeriesVAE.load_from_checkpoint(checkpoint_path)
    model.eval()

    dummy_input = torch.randn(1, 14, 50)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['reconstruction', 'mu', 'logvar'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'reconstruction': {0: 'batch_size'},
            'mu': {0: 'batch_size'},
            'logvar': {0: 'batch_size'}
        }
    )

    print(f"✅ Model exported to {output_path}")

if __name__ == "__main__":
    export_to_onnx('models/best_model.ckpt', 'models/model.onnx')
```

---

#### Person 1 & 2 - SUPPORT

**Priority Tasks:**
1. ✅ Write comprehensive tests (unit, integration, e2e)
2. ✅ Performance benchmarking
3. ✅ Complete documentation (API reference, model cards)
4. ✅ Create final presentation/demo

**Files to Create:**
- `tests/unit/test_*.py` (all modules)
- `tests/integration/test_training_pipeline.py`
- `tests/integration/test_inference_pipeline.py`
- `tests/e2e/test_api.py`
- `docs/api_reference.md`
- `docs/model_cards/*.md`
- `notebooks/05_final_demo.ipynb`

```python
# tests/unit/test_anomaly_detector.py
import pytest
import torch
from src.inference.anomaly_detector import AnomalyDetector

def test_anomaly_detector_predict():
    detector = AnomalyDetector('models/test_model.ckpt')
    x = torch.randn(10, 14, 50)
    scores, is_anomaly = detector.predict(x)

    assert len(scores) == 10
    assert len(is_anomaly) == 10
    assert all(isinstance(s, (float, np.float32, np.float64)) for s in scores)

def test_threshold_calibration():
    detector = AnomalyDetector('models/test_model.ckpt')
    # Test calibration logic...
    pass
```

```python
# tests/e2e/test_api.py
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_timeseries():
    data = {"data": [[0.1] * 50 for _ in range(14)]}
    response = client.post("/predict/timeseries", json=data)
    assert response.status_code == 200
    assert "anomaly_scores" in response.json()
```

---

### Week 4 Deliverables

**Completed by End of Week 4:**
- ✅ Multimodal VAE with PoE/MoE fusion
- ✅ Production REST API (FastAPI)
- ✅ ONNX model export
- ✅ Docker containerization
- ✅ Load balancing (Nginx)
- ✅ Monitoring (Prometheus + Grafana)
- ✅ Complete test suite (>80% coverage)
- ✅ Full documentation
- ✅ Final demo notebook

**Validation Criteria:**
- Multimodal VAE outperforms single-modality by >5% ROC-AUC
- API handles >100 req/sec
- Average latency < 50ms
- All tests passing
- Documentation complete
- Docker deployment working

---

## 📊 Complete Dependencies

```
# requirements/base.txt
torch>=2.1.0
pytorch-lightning>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.1.0
polars>=0.19.0
pyarrow>=14.0.0
librosa>=0.10.0
scikit-image>=0.22.0
PyWavelets>=1.5.0
scikit-learn>=1.3.0
shap>=0.43.0
captum>=0.7.0
wandb>=0.16.0
mlflow>=2.9.0
optuna>=3.4.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
python-multipart>=0.0.6
streamlit>=1.29.0
plotly>=5.18.0
matplotlib>=3.8.0
seaborn>=0.13.0
pytorch-msssim>=1.0.0
great-expectations>=0.18.0
hydra-core>=1.3.0
python-dotenv>=1.0.0
tqdm>=4.66.0
pillow>=10.1.0
onnx>=1.15.0
onnxruntime>=1.16.0

# requirements/dev.txt
-r base.txt
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-asyncio>=0.21.0
ruff>=0.1.0
mypy>=1.7.0
pre-commit>=3.5.0
jupyterlab>=4.0.0
ipywidgets>=8.1.0

# requirements/prod.txt
-r base.txt
gunicorn>=21.2.0
prometheus-client>=0.19.0
```

---

## 🚀 Getting Started

```bash
# 1. Setup
git clone <repo-url>
cd VAE-Anomaly-Detection
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements/dev.txt

# 2. Download datasets
bash scripts/download_datasets.sh

# 3. Train time-series VAE
python scripts/train.py

# 4. Evaluate
python scripts/evaluate.py --model_path models/best.ckpt --test_data data/test.txt

# 5. Run API
uvicorn src.api.main:app --reload

# 6. Run dashboard
streamlit run dashboards/streamlit_app.py

# 7. Docker deployment
docker-compose -f docker/docker-compose.yml up
```

---

## 📋 Weekly Milestones Checklist

### Week 1
- [ ] Project structure created
- [ ] NASA Turbofan data pipeline working
- [ ] Time-series VAE trained (val_loss < 0.15)
- [ ] Training dashboard operational
- [ ] Unit tests passing

### Week 2
- [ ] Anomaly detector implemented
- [ ] Threshold calibration working
- [ ] ROC-AUC > 0.85 on test set
- [ ] Root cause analyzer functional
- [ ] Interactive dashboards deployed

### Week 3
- [ ] Vision VAE trained (ROC-AUC > 0.90)
- [ ] Drift detector implemented
- [ ] Drift monitoring dashboard live
- [ ] Vision demo working
- [ ] All monitoring tests passing

### Week 4
- [ ] Multimodal VAE trained
- [ ] FastAPI deployed
- [ ] Docker containers working
- [ ] Load balancing configured
- [ ] Full test suite passing (>80% coverage)
- [ ] Documentation complete
- [ ] Final demo ready

---

## 🎯 Success Criteria

**Technical Metrics:**
- Time-series VAE: ROC-AUC > 0.85
- Vision VAE: ROC-AUC > 0.90
- Multimodal VAE: ROC-AUC > 0.92
- API latency: < 50ms (p95)
- Test coverage: > 80%

**Deliverables:**
- 3 trained VAE models (time-series, vision, multimodal)
- Production API with 5+ endpoints
- 5+ interactive Streamlit dashboards
- Docker-based deployment
- Comprehensive documentation
- Working demo notebook

**Engineering:**
- Modular, reusable codebase
- CI/CD pipeline functional
- Monitoring infrastructure deployed
- Drift detection automated
- Model versioning with DVC

---

This implementation plan provides a complete roadmap for building a production-grade multimodal anomaly detection system in 4 weeks with a 3-person team!
