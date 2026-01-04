# 🧠 PERSON 1: ML ENGINEER - Complete Task Guide
## 2-Day Implementation Plan

**Your Role:** Build ALL model architectures, training systems, and inference pipelines

**Total Time:** 16-20 hours per day
**Deliverables:** 3 trained VAE models + anomaly detection + root cause analysis

---

## 📋 Quick Task Overview

### DAY 1 (16-18 hours)
- **Hours 1-6:** Time-Series VAE + Vision VAE architectures + training
- **Hours 7-12:** Anomaly detection system + threshold calibration
- **Hours 13-18:** Root cause analysis + model optimization

### DAY 2 (16-18 hours)
- **Hours 1-8:** Multimodal VAE fusion + cross-modal attention
- **Hours 9-14:** Fine-tune all models + hyperparameter optimization
- **Hours 15-18:** Integration testing + final validation

---

## 🚀 DAY 1: CORE MODELS

### HOUR 1-3: Time-Series VAE Architecture

#### File: `src/models/encoders.py`

```python
import torch
import torch.nn as nn
from typing import Tuple

class Conv1DEncoder(nn.Module):
    """1D-CNN Encoder for time-series data

    Architecture:
    - Input: [batch, channels=14, seq_len=50]
    - 3 Conv1D layers with BatchNorm + LeakyReLU
    - Output: mu, logvar [batch, latent_dim]
    """

    def __init__(self,
                 input_channels: int,
                 sequence_length: int,
                 latent_dim: int = 128,
                 hidden_dims: list = [64, 128, 256]):
        super().__init__()

        self.latent_dim = latent_dim

        # Build encoder layers
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

        # Calculate flattened size
        self.flatten_size = self._get_flatten_size(input_channels, sequence_length)

        # Latent space projection
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def _get_flatten_size(self, input_channels: int, sequence_length: int) -> int:
        """Calculate size after convolutions"""
        x = torch.zeros(1, input_channels, sequence_length)
        x = self.encoder(x)
        return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, channels, sequence_length]
        Returns:
            mu: [batch_size, latent_dim]
            logvar: [batch_size, latent_dim]
        """
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class TransformerEncoder(nn.Module):
    """Transformer-based encoder (ALTERNATIVE - implement if time permits)

    More powerful but slower. Use for comparison.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, dim_feedforward)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward * 2,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc_mu = nn.Linear(dim_feedforward, latent_dim)
        self.fc_logvar = nn.Linear(dim_feedforward, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, sequence_length, features]
        """
        x = self.input_projection(x)
        h = self.transformer(x)
        h = h.mean(dim=1)  # Global average pooling

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
```

#### File: `src/models/decoders.py`

```python
import torch
import torch.nn as nn

class Conv1DDecoder(nn.Module):
    """1D-CNN Decoder for time-series reconstruction

    Mirror of encoder architecture
    """

    def __init__(self,
                 latent_dim: int,
                 output_channels: int,
                 sequence_length: int,
                 hidden_dims: list = [256, 128, 64]):
        super().__init__()

        self.sequence_length = sequence_length
        self.output_channels = output_channels

        # Calculate initial size after encoding
        self.initial_length = sequence_length // (2 ** len(hidden_dims))

        # Projection from latent
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * self.initial_length)

        # Build decoder layers
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i], hidden_dims[i+1],
                                      kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.LeakyReLU(0.2)
                )
            )

        # Final layer
        modules.append(
            nn.ConvTranspose1d(hidden_dims[-1], output_channels,
                              kernel_size=4, stride=2, padding=1)
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector [batch_size, latent_dim]
        Returns:
            Reconstructed sequence [batch_size, channels, sequence_length]
        """
        h = self.fc(z)
        h = h.view(-1, 256, self.initial_length)
        reconstruction = self.decoder(h)

        # Adjust to exact sequence length if needed
        if reconstruction.size(-1) != self.sequence_length:
            reconstruction = nn.functional.interpolate(
                reconstruction, size=self.sequence_length, mode='linear', align_corners=False
            )

        return reconstruction
```

#### File: `src/models/vae.py`

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Tuple
import wandb

class TimeSeriesVAE(pl.LightningModule):
    """Variational Autoencoder for Time-Series Anomaly Detection

    Key Features:
    - β-VAE formulation for better disentanglement
    - Separate reconstruction and KL losses
    - Anomaly scoring built-in
    - Lightning module for easy training
    """

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

        # Build encoder
        if encoder_type == 'cnn':
            from .encoders import Conv1DEncoder
            self.encoder = Conv1DEncoder(input_channels, sequence_length, latent_dim)
        elif encoder_type == 'transformer':
            from .encoders import TransformerEncoder
            self.encoder = TransformerEncoder(input_channels, latent_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Build decoder
        from .decoders import Conv1DDecoder
        self.decoder = Conv1DDecoder(latent_dim, input_channels, sequence_length)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for backprop through stochastic node"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Returns: (reconstruction, mu, logvar)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def compute_loss(self, x: torch.Tensor, reconstruction: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute VAE loss components

        Loss = Reconstruction Loss + β * KL Divergence
        """

        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='mean')

        # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss (β-VAE formulation)
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

        # Log metrics
        self.log_dict({f'train/{k}': v for k, v in losses.items()},
                     on_step=True, on_epoch=True, prog_bar=True)

        return losses['loss']

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, labels = batch
        reconstruction, mu, logvar = self(x)
        losses = self.compute_loss(x, reconstruction, mu, logvar)

        # Compute anomaly scores
        anomaly_scores = self.compute_anomaly_score(x, reconstruction, mu, logvar)

        # Log metrics
        self.log_dict({f'val/{k}': v for k, v in losses.items()},
                     on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': losses['loss'], 'anomaly_scores': anomaly_scores, 'labels': labels}

    def compute_anomaly_score(self, x: torch.Tensor, reconstruction: torch.Tensor,
                               mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score for each sample

        Score = Reconstruction Error + β * KL Divergence
        Higher score = more anomalous
        """

        # Per-sample reconstruction error
        recon_error = torch.mean((x - reconstruction) ** 2, dim=(1, 2))

        # Per-sample KL divergence
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # Combined anomaly score
        anomaly_score = recon_error + self.beta * kl_div

        return anomaly_score

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }
```

#### File: `scripts/train.py`

```python
#!/usr/bin/env python3
"""Training script for Time-Series VAE"""

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
import argparse

def train_timeseries_vae(config):
    """Train time-series VAE with given config"""

    # Setup W&B logger
    wandb_logger = WandbLogger(
        project="vae-anomaly-detection",
        name=f"timeseries-vae-{config['encoder_type']}",
        config=config
    )

    # Load data
    from src.data.loaders import TurbofanDataset

    full_dataset = TurbofanDataset(
        data_path=config['data_path'],
        sequence_length=config['sequence_length'],
        stride=config['stride']
    )

    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    from src.models.vae import TimeSeriesVAE

    model = TimeSeriesVAE(
        input_channels=config['input_channels'],
        sequence_length=config['sequence_length'],
        latent_dim=config['latent_dim'],
        encoder_type=config['encoder_type'],
        beta=config['beta'],
        learning_rate=config['learning_rate']
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"models/checkpoints/{wandb_logger.experiment.id}",
        filename="vae-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=10,
        mode="min",
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer
    trainer = Trainer(
        max_epochs=config['max_epochs'],
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        deterministic=True
    )

    # Train
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print(f"\n✅ Training complete!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

    return checkpoint_callback.best_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/raw/nasa_turbofan/train_FD001.txt')
    parser.add_argument('--encoder_type', default='cnn', choices=['cnn', 'transformer'])
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=1.0)

    args = parser.parse_args()

    config = {
        'data_path': args.data_path,
        'encoder_type': args.encoder_type,
        'input_channels': 14,  # Number of sensors
        'sequence_length': 50,
        'stride': 10,
        'latent_dim': args.latent_dim,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'learning_rate': args.learning_rate,
        'beta': args.beta
    }

    best_model_path = train_timeseries_vae(config)
```

**Run training:**
```bash
python scripts/train.py --max_epochs 30
```

---

### HOUR 4-6: Vision VAE Architecture

#### File: `src/models/vision_vae.py`

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple, Dict

class VisionEncoder(nn.Module):
    """2D-CNN Encoder for images (256x256 RGB)

    Architecture:
    - 5 Conv2D layers with stride=2 (downsampling)
    - Input: [B, 3, 256, 256]
    - Output: mu, logvar [B, latent_dim]
    """

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
    """2D-CNN Decoder for image reconstruction"""

    def __init__(self, latent_dim: int = 256, out_channels: int = 3):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)

        self.decoder = nn.Sequential(
            # [B, 512, 8, 8] -> [B, 512, 16, 16]
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # [B, 512, 16, 16] -> [B, 256, 32, 32]
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # [B, 256, 32, 32] -> [B, 128, 64, 64]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # [B, 128, 64, 64] -> [B, 64, 128, 128]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # [B, 64, 128, 128] -> [B, 3, 256, 256]
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 512, 8, 8)
        return self.decoder(h)


class VisionVAE(pl.LightningModule):
    """VAE for vision-based anomaly detection

    Uses SSIM loss for better perceptual reconstruction
    """

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

        # SSIM loss for perceptual quality
        from pytorch_msssim import ssim
        ssim_loss = 1 - ssim(reconstruction, x, data_range=1.0, size_average=True)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Combined loss
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

        # Log sample images
        if batch_idx == 0:
            self._log_images(x, reconstruction)

        return losses

    def _log_images(self, x, reconstruction):
        """Log sample reconstructions to W&B"""
        import torchvision
        import wandb

        # Take first 4 samples
        images = torch.cat([x[:4], reconstruction[:4]])
        grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)

        if self.logger:
            self.logger.experiment.log({
                "reconstructions": [wandb.Image(grid, caption="Top: Original, Bottom: Reconstructed")]
            })

    def compute_anomaly_score(self, x, reconstruction, mu, logvar):
        """Compute per-sample anomaly scores"""
        # Spatial reconstruction error
        recon_error = torch.mean((x - reconstruction) ** 2, dim=(1, 2, 3))

        # KL divergence
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        return recon_error + self.beta * kl_div

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val/loss'}
```

**Start vision training in parallel (Person 2 should have data ready):**
```bash
python scripts/train_vision.py --max_epochs 30
```

---

### HOUR 7-10: Anomaly Detection System

#### File: `src/inference/anomaly_detector.py`

```python
import torch
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score

class AnomalyDetector:
    """Anomaly detection using trained VAE

    Features:
    - Multiple threshold calibration methods
    - Evaluation metrics (ROC-AUC, PR-AUC, F1)
    - Save/load calibration
    """

    def __init__(self, model_path: str, threshold_percentile: float = 95.0, device: str = 'auto'):
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model.to(self.device)

        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.normal_scores_stats = None

    def _get_device(self, device: str):
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _load_model(self, path: str):
        """Load trained VAE model"""
        from src.models.vae import TimeSeriesVAE
        from src.models.vision_vae import VisionVAE

        # Try to load as time-series or vision VAE
        try:
            model = TimeSeriesVAE.load_from_checkpoint(path)
        except:
            model = VisionVAE.load_from_checkpoint(path)

        return model

    def calibrate_threshold(self, normal_data_loader, method: str = 'percentile'):
        """Calibrate anomaly threshold on normal data

        Args:
            normal_data_loader: DataLoader with normal (non-anomalous) samples
            method: 'percentile', 'mad', or 'gaussian'
        """
        scores = []

        print("Calibrating threshold on normal data...")
        with torch.no_grad():
            for batch_idx, (batch, _) in enumerate(normal_data_loader):
                batch = batch.to(self.device)
                batch_scores = self._compute_batch_scores(batch)
                scores.extend(batch_scores.cpu().numpy())

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {(batch_idx + 1) * normal_data_loader.batch_size} samples...")

        scores = np.array(scores)

        # Calculate threshold based on method
        from src.inference.threshold_calibrator import AdaptiveThresholdCalibrator
        calibrator = AdaptiveThresholdCalibrator()

        if method == 'percentile':
            self.threshold = calibrator.percentile_threshold(scores, self.threshold_percentile)
        elif method == 'mad':
            self.threshold = calibrator.mad_threshold(scores, n_sigma=3.0)
        elif method == 'gaussian':
            self.threshold = calibrator.gaussian_threshold(scores, n_sigma=3.0)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        # Store statistics
        self.normal_scores_stats = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'median': float(np.median(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'percentile_95': float(np.percentile(scores, 95)),
            'percentile_99': float(np.percentile(scores, 99)),
            'method': method
        }

        print(f"✅ Threshold calibrated: {self.threshold:.4f}")
        print(f"   Mean score: {self.normal_scores_stats['mean']:.4f}")
        print(f"   Std score: {self.normal_scores_stats['std']:.4f}")

        return self.threshold

    def _compute_batch_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores for a batch"""
        reconstruction, mu, logvar = self.model(x)
        scores = self.model.compute_anomaly_score(x, reconstruction, mu, logvar)
        return scores

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies

        Returns:
            scores: Anomaly scores for each sample
            is_anomaly: Boolean array indicating anomalies
        """
        if self.threshold is None:
            raise ValueError("Threshold not calibrated. Call calibrate_threshold() first.")

        x = x.to(self.device)

        with torch.no_grad():
            scores = self._compute_batch_scores(x).cpu().numpy()

        is_anomaly = scores > self.threshold

        return scores, is_anomaly

    def evaluate(self, test_loader) -> Dict[str, float]:
        """Evaluate on test set with labels

        Returns:
            Dictionary with metrics: ROC-AUC, PR-AUC, Precision, Recall, F1
        """
        all_scores = []
        all_labels = []

        print("Evaluating model on test set...")
        with torch.no_grad():
            for batch_idx, (batch, labels) in enumerate(test_loader):
                batch = batch.to(self.device)
                scores = self._compute_batch_scores(batch)
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {(batch_idx + 1) * test_loader.batch_size} samples...")

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # ROC-AUC
        roc_auc = roc_auc_score(all_labels, all_scores)

        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_scores)
        pr_auc = auc(recall_curve, precision_curve)

        # Precision, Recall, F1 at calibrated threshold
        predictions = (all_scores > self.threshold).astype(int)

        tp = np.sum((predictions == 1) & (all_labels == 1))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        fn = np.sum((predictions == 0) & (all_labels == 1))
        tn = np.sum((predictions == 0) & (all_labels == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        accuracy = (tp + tn) / len(all_labels)

        metrics = {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'threshold': float(self.threshold),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }

        print(f"\n{'='*60}")
        print("EVALUATION RESULTS:")
        print(f"{'='*60}")
        for k, v in metrics.items():
            if k not in ['tp', 'fp', 'fn', 'tn']:
                print(f"{k:20s}: {v:.4f}")
        print(f"{'='*60}\n")

        return metrics

    def save_calibration(self, path: str):
        """Save threshold and statistics"""
        calibration_data = {
            'threshold': self.threshold,
            'threshold_percentile': self.threshold_percentile,
            'normal_scores_stats': self.normal_scores_stats
        }

        with open(path, 'wb') as f:
            pickle.dump(calibration_data, f)

        print(f"✅ Calibration saved to {path}")

    def load_calibration(self, path: str):
        """Load threshold and statistics"""
        with open(path, 'rb') as f:
            calibration_data = pickle.load(f)

        self.threshold = calibration_data['threshold']
        self.threshold_percentile = calibration_data.get('threshold_percentile', 95.0)
        self.normal_scores_stats = calibration_data['normal_scores_stats']

        print(f"✅ Calibration loaded from {path}")
        print(f"   Threshold: {self.threshold:.4f}")
```

#### File: `src/inference/threshold_calibrator.py`

```python
import numpy as np
from scipy import stats
from typing import Optional

class AdaptiveThresholdCalibrator:
    """Advanced threshold calibration strategies

    Methods:
    1. Percentile: Simple percentile-based
    2. MAD: Median Absolute Deviation (robust to outliers)
    3. Gaussian: Assumes normal distribution
    4. EVT: Extreme Value Theory
    5. Dynamic: For streaming data
    """

    @staticmethod
    def percentile_threshold(scores: np.ndarray, percentile: float = 95) -> float:
        """Simple percentile-based threshold

        Args:
            scores: Anomaly scores from normal data
            percentile: Percentile to use (95 = 95th percentile)
        """
        return float(np.percentile(scores, percentile))

    @staticmethod
    def mad_threshold(scores: np.ndarray, n_sigma: float = 3.0) -> float:
        """Median Absolute Deviation threshold (robust to outliers)

        MAD is more robust than standard deviation
        """
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        # Scaling factor 0.6745 converts MAD to standard deviation equivalent
        threshold = median + n_sigma * mad / 0.6745
        return float(threshold)

    @staticmethod
    def gaussian_threshold(scores: np.ndarray, n_sigma: float = 3.0) -> float:
        """Gaussian assumption threshold

        Assumes scores follow normal distribution
        """
        mean = np.mean(scores)
        std = np.std(scores)
        return float(mean + n_sigma * std)

    @staticmethod
    def extreme_value_threshold(scores: np.ndarray, quantile: float = 0.95) -> float:
        """Extreme Value Theory (EVT) based threshold

        Fits Generalized Extreme Value distribution
        Better for tail probabilities
        """
        try:
            params = stats.genextreme.fit(scores)
            threshold = stats.genextreme.ppf(quantile, *params)
            return float(threshold)
        except:
            # Fallback to percentile if fitting fails
            return AdaptiveThresholdCalibrator.percentile_threshold(scores, quantile * 100)

    @staticmethod
    def dynamic_threshold(scores: np.ndarray, window_size: int = 100,
                         n_sigma: float = 3.0) -> np.ndarray:
        """Dynamic threshold for streaming data

        Computes threshold based on sliding window
        """
        thresholds = []

        for i in range(len(scores)):
            start = max(0, i - window_size)
            window = scores[start:i+1]

            if len(window) < 10:
                thresholds.append(np.inf)
            else:
                mean = np.mean(window)
                std = np.std(window)
                thresholds.append(mean + n_sigma * std)

        return np.array(thresholds)
```

#### File: `scripts/evaluate.py`

```python
#!/usr/bin/env python3
"""Evaluation script for trained VAE models"""

import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path
import argparse

def evaluate_model(model_path: str, test_data_path: str, output_dir: str, model_type: str = 'timeseries'):
    """Comprehensive model evaluation"""

    from src.inference.anomaly_detector import AnomalyDetector

    # Load test data
    if model_type == 'timeseries':
        from src.data.loaders import TurbofanDataset
        test_dataset = TurbofanDataset(test_data_path, sequence_length=50)
    elif model_type == 'vision':
        from src.data.vision_loaders import MVTecDataset
        test_dataset = MVTecDataset(root_dir=test_data_path, split='test')

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Initialize detector
    detector = AnomalyDetector(model_path, threshold_percentile=95)

    # Calibrate on normal samples (first 70% assumed normal)
    normal_size = int(len(test_dataset) * 0.7)
    normal_dataset = torch.utils.data.Subset(test_dataset, range(normal_size))
    normal_loader = DataLoader(normal_dataset, batch_size=128, num_workers=4)

    print("=" * 60)
    print("CALIBRATING THRESHOLD")
    print("=" * 60)
    detector.calibrate_threshold(normal_loader, method='percentile')

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    metrics = detector.evaluate(test_loader)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save calibration
    detector.save_calibration(output_path / 'calibration.pkl')

    print(f"\n✅ Results saved to {output_dir}/")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to model checkpoint')
    parser.add_argument('--test_data', required=True, help='Path to test data')
    parser.add_argument('--output_dir', default='evaluation_results', help='Output directory')
    parser.add_argument('--model_type', default='timeseries', choices=['timeseries', 'vision'])

    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_data, args.output_dir, args.model_type)
```

---

### HOUR 11-14: Root Cause Analysis

#### File: `src/inference/root_cause_analyzer.py`

```python
import torch
import numpy as np
from typing import Dict, List, Optional
import shap

class RootCauseAnalyzer:
    """Root cause analysis for detected anomalies

    Features:
    - Per-sensor reconstruction error
    - Temporal contribution analysis
    - Latent sensitivity analysis
    - SHAP-based explainability (optional)
    """

    def __init__(self, model, sensor_names: List[str], device: str = 'auto'):
        self.model = model
        self.model.eval()
        self.sensor_names = sensor_names

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

    def analyze_reconstruction_error(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Compute per-sensor reconstruction error

        Identifies which sensors contribute most to anomaly
        """
        x = x.to(self.device)

        with torch.no_grad():
            reconstruction, mu, logvar = self.model(x)

        # Per-sensor error [batch, sensors, time]
        error = (x - reconstruction) ** 2

        # Aggregate over time dimension
        sensor_errors = error.mean(dim=2).cpu().numpy()  # [batch, sensors]

        # Average over batch
        avg_sensor_errors = sensor_errors.mean(axis=0)  # [sensors]

        # Rank sensors by error
        sensor_ranking = np.argsort(avg_sensor_errors)[::-1]  # Descending order

        return {
            'sensor_errors': sensor_errors,
            'avg_sensor_errors': avg_sensor_errors,
            'sensor_ranking': sensor_ranking,
            'top_sensors': [self.sensor_names[i] for i in sensor_ranking[:5]],
            'top_sensor_scores': avg_sensor_errors[sensor_ranking[:5]]
        }

    def analyze_temporal_contribution(self, x: torch.Tensor,
                                     window_size: int = 10) -> Dict[str, np.ndarray]:
        """Analyze which time windows contribute most to anomaly

        Identifies when the anomaly occurs in the sequence
        """
        x = x.to(self.device)

        with torch.no_grad():
            reconstruction, mu, logvar = self.model(x)

        error = (x - reconstruction) ** 2  # [batch, sensors, time]

        # Sliding window error
        seq_length = error.shape[2]
        num_windows = seq_length - window_size + 1

        window_errors = []
        for i in range(num_windows):
            window_error = error[:, :, i:i+window_size].mean(dim=(1, 2))
            window_errors.append(window_error)

        window_errors = torch.stack(window_errors, dim=1).cpu().numpy()  # [batch, windows]

        # Find peak window
        peak_window_idx = np.argmax(window_errors.mean(axis=0))

        return {
            'window_errors': window_errors,
            'avg_window_errors': window_errors.mean(axis=0),
            'peak_window_idx': int(peak_window_idx),
            'peak_window_start': int(peak_window_idx),
            'peak_window_end': int(peak_window_idx + window_size),
            'window_size': window_size
        }

    def analyze_latent_sensitivity(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Analyze sensitivity of reconstruction to latent dimensions

        Identifies which latent dimensions are most important
        """
        x = x.to(self.device)

        # Get latent representation
        with torch.no_grad():
            mu, logvar = self.model.encoder(x)
            z = self.model.reparameterize(mu, logvar)

        # Compute gradient of reconstruction w.r.t. latent
        z_copy = z.clone().detach().requires_grad_(True)
        reconstruction = self.model.decoder(z_copy)

        # Sensitivity: ||∂reconstruction/∂z||
        sensitivities = []

        for i in range(x.shape[0]):  # For each sample
            grad_outputs = torch.ones_like(reconstruction[i:i+1])
            grads = torch.autograd.grad(
                outputs=reconstruction[i:i+1],
                inputs=z_copy,
                grad_outputs=grad_outputs,
                create_graph=False
            )[0]

            # L2 norm of gradients per latent dimension
            sensitivity = torch.norm(grads[i], p=2).item()
            sensitivities.append(sensitivity)

        sensitivities = np.array(sensitivities)

        # Most sensitive dimensions
        latent_importance = np.abs(z.mean(dim=0).cpu().numpy())
        most_sensitive_dims = np.argsort(latent_importance)[::-1][:10]

        return {
            'sensitivities': sensitivities,
            'latent_importance': latent_importance,
            'most_sensitive_dims': most_sensitive_dims.tolist(),
            'top_latent_values': latent_importance[most_sensitive_dims].tolist()
        }

    def generate_report(self, x: torch.Tensor, anomaly_idx: int) -> Dict:
        """Generate comprehensive root cause report for a single anomaly

        Args:
            x: Batch of samples
            anomaly_idx: Index of anomalous sample in batch

        Returns:
            Dictionary with detailed root cause analysis
        """
        sample = x[anomaly_idx:anomaly_idx+1]

        print(f"Analyzing anomaly at index {anomaly_idx}...")

        # Reconstruction error analysis
        recon_analysis = self.analyze_reconstruction_error(sample)

        # Temporal analysis
        temporal_analysis = self.analyze_temporal_contribution(sample)

        # Latent sensitivity
        latent_analysis = self.analyze_latent_sensitivity(sample)

        # Build comprehensive report
        report = {
            'anomaly_index': anomaly_idx,

            # Sensor analysis
            'top_contributing_sensors': recon_analysis['top_sensors'],
            'sensor_error_scores': {
                self.sensor_names[i]: float(recon_analysis['avg_sensor_errors'][i])
                for i in range(len(self.sensor_names))
            },
            'sensor_ranking': [self.sensor_names[i] for i in recon_analysis['sensor_ranking']],

            # Temporal analysis
            'peak_time_window': temporal_analysis['peak_window_idx'],
            'peak_window_start': temporal_analysis['peak_window_start'],
            'peak_window_end': temporal_analysis['peak_window_end'],

            # Latent analysis
            'most_sensitive_latent_dims': latent_analysis['most_sensitive_dims'],
            'latent_importance_scores': latent_analysis['top_latent_values']
        }

        return report

    def batch_analyze(self, x: torch.Tensor, anomaly_indices: List[int]) -> List[Dict]:
        """Generate reports for multiple anomalies"""
        reports = []

        for idx in anomaly_indices:
            report = self.generate_report(x, idx)
            reports.append(report)

        return reports
```

**Test root cause analysis:**
```python
# In notebook or script
from src.models.vae import TimeSeriesVAE
from src.inference.root_cause_analyzer import RootCauseAnalyzer
import torch

model = TimeSeriesVAE.load_from_checkpoint('models/best_model.ckpt')
sensor_names = [f'Sensor_{i}' for i in range(1, 15)]
analyzer = RootCauseAnalyzer(model, sensor_names)

# Analyze sample
x = torch.randn(10, 14, 50)  # Replace with real data
report = analyzer.generate_report(x, anomaly_idx=0)
print(report)
```

---

## 🚀 DAY 2: MULTIMODAL FUSION & OPTIMIZATION

### HOUR 1-8: Multimodal VAE

#### File: `src/models/multimodal_vae.py`

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Tuple, Optional

class MultimodalVAE(pl.LightningModule):
    """Multimodal VAE with Product-of-Experts (PoE) fusion

    Combines time-series and vision modalities into shared latent space

    Fusion Methods:
    1. Product-of-Experts (PoE): Precision-weighted fusion
    2. Mixture-of-Experts (MoE): Simple averaging
    3. Concatenation: Direct concatenation
    """

    def __init__(self,
                 timeseries_encoder,
                 vision_encoder,
                 shared_latent_dim: int = 256,
                 fusion_method: str = 'poe',
                 beta: float = 1.0,
                 learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['timeseries_encoder', 'vision_encoder'])

        # Modality-specific encoders
        self.ts_encoder = timeseries_encoder
        self.vis_encoder = vision_encoder

        self.shared_latent_dim = shared_latent_dim
        self.fusion_method = fusion_method

        # Cross-modal attention (optional enhancement)
        self.use_attention = True
        if self.use_attention:
            self.cross_attention = CrossModalAttention(shared_latent_dim)

        # Modality-specific decoders
        from .decoders import Conv1DDecoder
        from .vision_vae import VisionDecoder

        self.ts_decoder = Conv1DDecoder(shared_latent_dim, 14, 50)
        self.vis_decoder = VisionDecoder(shared_latent_dim, 3)

        self.beta = beta
        self.learning_rate = learning_rate

    def encode(self, ts_data: Optional[torch.Tensor] = None,
               vis_data: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode multimodal data into shared latent space"""

        mus, logvars = [], []

        # Encode time-series
        if ts_data is not None:
            ts_mu, ts_logvar = self.ts_encoder(ts_data)
            mus.append(ts_mu)
            logvars.append(ts_logvar)

        # Encode vision
        if vis_data is not None:
            vis_mu, vis_logvar = self.vis_encoder(vis_data)
            mus.append(vis_mu)
            logvars.append(vis_logvar)

        # Fusion
        if len(mus) == 0:
            raise ValueError("At least one modality must be provided")
        elif len(mus) == 1:
            mu, logvar = mus[0], logvars[0]
        else:
            if self.fusion_method == 'poe':
                mu, logvar = self._product_of_experts(mus, logvars)
            elif self.fusion_method == 'moe':
                mu, logvar = self._mixture_of_experts(mus, logvars)
            elif self.fusion_method == 'concat':
                mu = torch.cat(mus, dim=-1)
                logvar = torch.cat(logvars, dim=-1)
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return mu, logvar

    def _product_of_experts(self, mus: list, logvars: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Product of Experts fusion (precision-weighted)

        More weight to modalities with higher confidence (lower variance)
        """
        # Convert variance to precision (1/variance)
        precisions = [torch.exp(-lv) for lv in logvars]
        precision_sum = sum(precisions)

        # Precision-weighted mean
        mu = sum([p * m for p, m in zip(precisions, mus)]) / (precision_sum + 1e-8)

        # Combined variance
        var = 1.0 / (precision_sum + 1e-8)
        logvar = torch.log(var + 1e-8)

        return mu, logvar

    def _mixture_of_experts(self, mus: list, logvars: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixture of Experts fusion (simple averaging)"""
        mu = torch.stack(mus).mean(dim=0)
        logvar = torch.stack(logvars).mean(dim=0)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode latent into both modalities"""
        ts_recon = self.ts_decoder(z)
        vis_recon = self.vis_decoder(z)

        return {
            'timeseries': ts_recon,
            'vision': vis_recon
        }

    def forward(self, ts_data: Optional[torch.Tensor] = None,
                vis_data: Optional[torch.Tensor] = None):
        """Forward pass"""

        # Encode
        mu, logvar = self.encode(ts_data, vis_data)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Apply cross-attention if enabled
        if self.use_attention and ts_data is not None and vis_data is not None:
            # Get separate encodings
            ts_mu, _ = self.ts_encoder(ts_data)
            vis_mu, _ = self.vis_encoder(vis_data)

            # Cross-attend
            z = self.cross_attention(z, ts_mu, vis_mu)

        # Decode
        reconstructions = self.decode(z)

        return reconstructions, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_loss(self, ts_data, vis_data, reconstructions, mu, logvar):
        """Compute multimodal loss"""
        losses = {}

        # Time-series reconstruction
        if ts_data is not None:
            ts_recon_loss = nn.functional.mse_loss(reconstructions['timeseries'], ts_data)
            losses['ts_recon'] = ts_recon_loss

        # Vision reconstruction
        if vis_data is not None:
            vis_recon_loss = nn.functional.mse_loss(reconstructions['vision'], vis_data)
            losses['vis_recon'] = vis_recon_loss

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        losses['kl'] = kl_loss

        # Total loss
        total = sum(v for k, v in losses.items() if 'recon' in k) + self.beta * kl_loss
        losses['loss'] = total

        return losses

    def training_step(self, batch, batch_idx):
        ts_data, vis_data, _ = batch

        reconstructions, mu, logvar = self(ts_data, vis_data)
        losses = self.compute_loss(ts_data, vis_data, reconstructions, mu, logvar)

        self.log_dict({f'train/{k}': v for k, v in losses.items()}, prog_bar=True)
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        ts_data, vis_data, labels = batch

        reconstructions, mu, logvar = self(ts_data, vis_data)
        losses = self.compute_loss(ts_data, vis_data, reconstructions, mu, logvar)

        self.log_dict({f'val/{k}': v for k, v in losses.items()}, prog_bar=True)
        return losses

    def compute_anomaly_score(self, ts_data, vis_data, reconstructions, mu, logvar):
        """Compute multimodal anomaly score"""
        scores = []

        # Time-series error
        if ts_data is not None:
            ts_error = torch.mean((ts_data - reconstructions['timeseries']) ** 2, dim=(1, 2))
            scores.append(ts_error)

        # Vision error
        if vis_data is not None:
            vis_error = torch.mean((vis_data - reconstructions['vision']) ** 2, dim=(1, 2, 3))
            scores.append(vis_error)

        # KL divergence
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        scores.append(kl_div)

        # Weighted average (can adjust weights)
        total_score = sum(scores) / len(scores)

        return total_score

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism

    Allows information flow between modalities
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()

        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, fused: torch.Tensor, ts_feat: torch.Tensor, vis_feat: torch.Tensor):
        """
        Args:
            fused: Fused representation [batch, dim]
            ts_feat: Time-series features [batch, dim]
            vis_feat: Vision features [batch, dim]
        """
        # Add sequence dimension
        fused = fused.unsqueeze(1)  # [batch, 1, dim]

        # Stack modality features as key/value
        kv = torch.stack([ts_feat, vis_feat], dim=1)  # [batch, 2, dim]

        # Cross-attention
        attn_out, _ = self.attention(fused, kv, kv)
        attn_out = attn_out.squeeze(1)  # [batch, dim]

        # Residual + norm
        out = self.norm(fused.squeeze(1) + attn_out)

        return out
```

#### File: `scripts/train_multimodal.py`

```python
#!/usr/bin/env python3
"""Train multimodal VAE"""

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

def train_multimodal_vae():
    # Load pre-trained encoders
    from src.models.vae import TimeSeriesVAE
    from src.models.vision_vae import VisionVAE

    ts_vae = TimeSeriesVAE.load_from_checkpoint('models/timeseries_best.ckpt')
    vis_vae = VisionVAE.load_from_checkpoint('models/vision_best.ckpt')

    # Extract encoders
    ts_encoder = ts_vae.encoder
    vis_encoder = vis_vae.encoder

    # Create multimodal VAE
    from src.models.multimodal_vae import MultimodalVAE

    model = MultimodalVAE(
        timeseries_encoder=ts_encoder,
        vision_encoder=vis_encoder,
        shared_latent_dim=256,
        fusion_method='poe',
        beta=1.0,
        learning_rate=1e-4
    )

    # Load multimodal dataset
    from src.data.multimodal_loader import MultimodalDataset
    from src.data.loaders import TurbofanDataset
    from src.data.vision_loaders import MVTecDataset

    ts_dataset = TurbofanDataset('data/raw/nasa_turbofan/train_FD001.txt')
    vis_dataset = MVTecDataset('data/raw/mvtec', category='bottle', split='train')

    multimodal_dataset = MultimodalDataset(ts_dataset, vis_dataset)

    train_loader = DataLoader(multimodal_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Logger
    logger = WandbLogger(project="vae-anomaly-detection", name="multimodal-vae")

    # Callbacks
    checkpoint = ModelCheckpoint(monitor='val/loss', save_top_k=3)
    early_stop = EarlyStopping(monitor='val/loss', patience=10)

    # Trainer
    trainer = Trainer(
        max_epochs=30,
        accelerator='auto',
        logger=logger,
        callbacks=[checkpoint, early_stop]
    )

    # Train
    trainer.fit(model, train_loader)

    print(f"✅ Multimodal VAE trained! Best model: {checkpoint.best_model_path}")

if __name__ == "__main__":
    train_multimodal_vae()
```

---

### HOUR 9-14: Hyperparameter Optimization & Fine-tuning

#### File: `scripts/hyperparameter_optimization.py`

```python
#!/usr/bin/env python3
"""Hyperparameter optimization using Optuna"""

import optuna
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

def objective(trial):
    """Optuna objective function"""

    # Suggest hyperparameters
    config = {
        'latent_dim': trial.suggest_int('latent_dim', 64, 256, step=64),
        'beta': trial.suggest_float('beta', 0.5, 2.0),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3)
    }

    # Train model with these hyperparameters
    from src.models.vae import TimeSeriesVAE
    from src.data.loaders import TurbofanDataset
    from torch.utils.data import DataLoader

    dataset = TurbofanDataset('data/raw/nasa_turbofan/train_FD001.txt')
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = TimeSeriesVAE(
        input_channels=14,
        sequence_length=50,
        latent_dim=config['latent_dim'],
        beta=config['beta'],
        learning_rate=config['learning_rate']
    )

    trainer = Trainer(max_epochs=20, accelerator='auto', enable_checkpointing=False)
    trainer.fit(model, train_loader)

    # Return validation loss
    return trainer.callback_metrics['val/loss'].item()

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    print(f"✅ Best hyperparameters: {study.best_params}")
    print(f"   Best loss: {study.best_value:.4f}")
```

---

### HOUR 15-18: Integration & Final Validation

Create comprehensive test script and final validation notebook.

**Quick validation:**
```bash
# Test all models
python scripts/validate_all_models.py

# Generate demo notebook
jupyter nbconvert --execute notebooks/final_demo.ipynb
```

---

## ✅ Your Deliverables Checklist

### Models
- [ ] Time-Series VAE trained (ROC-AUC > 0.80)
- [ ] Vision VAE trained (ROC-AUC > 0.85)
- [ ] Multimodal VAE trained (ROC-AUC > 0.88)
- [ ] All models saved and versioned

### Inference Systems
- [ ] Anomaly detector with calibrated thresholds
- [ ] Root cause analyzer working
- [ ] Evaluation pipeline complete
- [ ] Multiple threshold methods implemented

### Code Quality
- [ ] All model files created
- [ ] Training scripts working
- [ ] Unit tests passing
- [ ] Documentation complete

---

## 🚨 Troubleshooting

**GPU Out of Memory:**
```python
# Reduce batch size
config['batch_size'] = 64  # or 32

# Use gradient accumulation
trainer = Trainer(accumulate_grad_batches=2)
```

**Training Too Slow:**
```python
# Reduce epochs for initial training
config['max_epochs'] = 20

# Use lighter architecture
config['hidden_dims'] = [32, 64, 128]  # Instead of [64, 128, 256]
```

**Models Not Converging:**
```python
# Adjust learning rate
config['learning_rate'] = 5e-4

# Increase beta gradually
config['beta'] = 0.5  # Start lower
```

---

## 📞 Sync Points with Team

**After 6 hours:**
- Share trained time-series VAE checkpoint with Person 3
- Confirm data loaders working with Person 2
- Provide anomaly scores for dashboard testing

**End of Day 1:**
- Commit all model code
- Share evaluation results
- Provide model checkpoints

**After 6 hours Day 2:**
- Share multimodal VAE
- Provide all three models for API integration
- Final testing with Person 3

---

**You got this! Focus on getting models trained first, then polish. Quality over perfection in 2 days! 🚀**
