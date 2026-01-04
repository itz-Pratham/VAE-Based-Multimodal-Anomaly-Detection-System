# ⚡ QUICK START GUIDE
## Get Running in 30 Minutes

**For all team members to start immediately!**

---

## 🚀 Installation (Everyone - Do This First!)

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd VAE-Based-Multimodal-Anomaly-Detection-System
```

### Step 2: Create Virtual Environment

```bash
# Create venv
python3.10 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all requirements
pip install --upgrade pip
pip install -r requirements/dev.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
```

**Expected output:**
```
PyTorch: 2.1.0+cpu (or cu118 for GPU)
Streamlit: 1.29.0
```

---

## 📥 Download Datasets (Person 2 Priority, Others Can Start Later)

```bash
# Make script executable
chmod +x scripts/download_datasets.sh

# Download all datasets (~10GB, takes 30-60 min)
./scripts/download_datasets.sh
```

**Alternative - Manual Download:**

1. **NASA Turbofan:** https://ti.arc.nasa.gov/c/6/
2. **MIMII Sound:** https://zenodo.org/record/3384388
3. **MVTec AD:** https://www.mvtec.com/company/research/datasets/mvtec-ad

Extract to:
- `data/raw/nasa_turbofan/`
- `data/raw/mimii/`
- `data/raw/mvtec/`

---

## 🧪 Verify Setup

```bash
# Run tests
pytest tests/unit/ -v

# Expected: All tests pass or skip (if data not downloaded yet)
```

---

## 🎯 Person-Specific Quick Starts

### Person 1 (ML Engineer)

**Start Training Immediately:**

```bash
# 1. Create a test data loader (mock data if datasets not ready)
python -c "from src.data.loaders import TurbofanDataset; print('✅ Loaders work!')"

# 2. Test model creation
python -c "from src.models.vae import TimeSeriesVAE; model = TimeSeriesVAE(14, 50); print('✅ Model created!')"

# 3. Start training (use subset if full data not ready)
python scripts/train.py --max_epochs 5 --batch_size 64

# 4. Setup W&B
wandb login
# Paste your API key from: https://wandb.ai/authorize
```

**Your First Hour:**
1. ✅ Verify model imports work
2. ✅ Start training on small subset
3. ✅ Setup W&B tracking
4. ✅ Read `PERSON1_ML_ENGINEER.md`

---

### Person 2 (Data Engineer)

**Start Data Pipeline:**

```bash
# 1. Test data loaders
cd tests/unit
pytest test_loaders.py -v

# 2. Start dataset download (background)
nohup ./scripts/download_datasets.sh > download.log 2>&1 &

# 3. Setup DVC
dvc init
git add .dvc/config

# 4. Setup W&B/MLflow
pip install wandb mlflow
wandb login
```

**Your First Hour:**
1. ✅ Start dataset download
2. ✅ Test data loader code
3. ✅ Setup DVC
4. ✅ Read `PERSON2_DATA_ENGINEER.md`

---

### Person 3 (Full-Stack)

**Start Dashboard Development:**

```bash
# 1. Test Streamlit
streamlit hello

# 2. Run main dashboard
streamlit run dashboards/streamlit_app.py

# 3. Test Docker (if installed)
docker --version
docker-compose --version

# 4. Start building dashboards
code dashboards/streamlit_app.py
```

**Your First Hour:**
1. ✅ Verify Streamlit works
2. ✅ Test Docker/Docker Compose
3. ✅ Start dashboard skeleton
4. ✅ Read `PERSON3_FULLSTACK.md`

---

## 🔧 Common Setup Issues & Fixes

### Issue: `ModuleNotFoundError`

```bash
# Fix: Install package in editable mode
pip install -e .
```

### Issue: PyTorch CUDA not found (GPU users)

```bash
# Fix: Install GPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Streamlit won't start

```bash
# Fix: Check port
lsof -i :8501
# Kill if needed: kill -9 <PID>

# Or use different port
streamlit run dashboards/streamlit_app.py --server.port 8502
```

### Issue: Docker permission denied

```bash
# Fix (Linux/macOS):
sudo usermod -aG docker $USER
# Then logout and login
```

### Issue: Out of memory during training

```bash
# Fix: Reduce batch size
python scripts/train.py --batch_size 32
# Or use gradient accumulation
```

---

## 📁 Project Structure Overview

```
VAE-Anomaly-Detection/
├── src/                    # All source code
│   ├── data/              # Data loaders & preprocessing
│   ├── models/            # VAE architectures
│   ├── inference/         # Anomaly detection & root cause
│   ├── monitoring/        # Drift detection
│   └── api/               # FastAPI service
├── dashboards/            # Streamlit dashboards
├── scripts/               # Training & utility scripts
├── tests/                 # Unit & integration tests
├── docker/                # Docker configs
├── data/                  # Datasets (gitignored)
├── models/                # Trained models (gitignored)
└── requirements/          # Dependencies
```

---

## 🚦 Getting Started Workflow

### Hour 0-1: Setup

```bash
# Everyone
git clone <repo>
cd VAE-Anomaly-Detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/dev.txt

# Person 2
./scripts/download_datasets.sh &

# Person 1
wandb login

# Person 3
streamlit run dashboards/streamlit_app.py
```

### Hour 1-4: Core Development

**Person 1:**
```bash
# Create models
cd src/models
# Edit vae.py, encoders.py, decoders.py

# Test
python -c "from models.vae import TimeSeriesVAE; print('✅')"

# Start training
python scripts/train.py
```

**Person 2:**
```bash
# Create loaders
cd src/data
# Edit loaders.py, feature_extractors.py

# Test
pytest tests/unit/test_loaders.py

# Monitor dataset download
tail -f download.log
```

**Person 3:**
```bash
# Build dashboards
cd dashboards
# Edit streamlit_app.py

# Test
streamlit run streamlit_app.py

# Build Docker
docker build -f docker/Dockerfile.dashboard -t vae-dashboard .
```

### Hour 4-8: Integration

**Sync Point 1:**
- Person 2 → Person 1: Confirm data loaders ready
- Person 1 → Person 3: Share model checkpoints
- Person 3 → All: Share dashboard URL

**Continue Development:**
```bash
# Person 1: Continue training
python scripts/train.py --max_epochs 50

# Person 2: Start evaluation pipeline
python scripts/evaluate.py --model_path models/best.ckpt

# Person 3: Build API
cd src/api
# Edit main.py
```

---

## 🧪 Testing Your Setup

### Quick Smoke Test

```bash
# Run this to verify everything works
python << 'EOF'
import torch
import numpy as np
from src.data.loaders import TurbofanDataset
from src.models.vae import TimeSeriesVAE

print("Testing data loader...")
# Mock data if dataset not available
x = torch.randn(10, 14, 50)
print(f"✅ Data shape: {x.shape}")

print("\nTesting model...")
model = TimeSeriesVAE(14, 50, latent_dim=128)
recon, mu, logvar = model(x)
print(f"✅ Model output: {recon.shape}")

print("\nTesting loss computation...")
losses = model.compute_loss(x, recon, mu, logvar)
print(f"✅ Loss: {losses['loss']:.4f}")

print("\n🎉 ALL TESTS PASSED!")
EOF
```

Expected output:
```
Testing data loader...
✅ Data shape: torch.Size([10, 14, 50])

Testing model...
✅ Model output: torch.Size([10, 14, 50])

Testing loss computation...
✅ Loss: 1.2345

🎉 ALL TESTS PASSED!
```

---

## 📚 Essential Commands Cheatsheet

### Git Commands

```bash
# Create feature branch
git checkout -b feature/your-feature

# Commit changes
git add .
git commit -m "feat: your feature"

# Push
git push origin feature/your-feature

# Pull latest
git pull origin main
```

### Training Commands

```bash
# Train time-series VAE
python scripts/train.py

# Train with custom config
python scripts/train.py --latent_dim 256 --batch_size 128 --max_epochs 50

# Resume training
python scripts/train.py --resume_from models/checkpoints/last.ckpt
```

### Evaluation Commands

```bash
# Evaluate model
python scripts/evaluate.py \
    --model_path models/best_model.ckpt \
    --test_data data/raw/nasa_turbofan/test_FD001.txt

# Export to ONNX
python scripts/export_onnx.py \
    --checkpoint models/best_model.ckpt \
    --output models/model.onnx
```

### Dashboard Commands

```bash
# Run main dashboard
streamlit run dashboards/streamlit_app.py

# Run specific page
streamlit run dashboards/pages/1_training_monitor.py

# Run on different port
streamlit run dashboards/streamlit_app.py --server.port 8502
```

### Docker Commands

```bash
# Build all images
docker-compose -f docker/docker-compose.yml build

# Start all services
docker-compose -f docker/docker-compose.yml up

# Start in background
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose logs -f api

# Stop all
docker-compose down

# Rebuild and start
docker-compose up --build
```

### Testing Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_loaders.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run and show print statements
pytest -v -s
```

---

## 🎯 Success Checklist

After setup, you should have:

### Person 1 (ML Engineer)
- [ ] Can create and run VAE models
- [ ] W&B tracking configured
- [ ] Training script works
- [ ] Can see training metrics

### Person 2 (Data Engineer)
- [ ] Data loaders import successfully
- [ ] Dataset download started/completed
- [ ] DVC initialized
- [ ] Tests pass

### Person 3 (Full-Stack)
- [ ] Streamlit dashboard runs
- [ ] Docker images build
- [ ] Can access dashboard in browser
- [ ] FastAPI skeleton created

---

## 🆘 Get Help

**If stuck for >15 minutes:**

1. **Check documentation:**
   - Your role-specific guide (`PERSON{1,2,3}*.md`)
   - `MASTER_PLAN.md`

2. **Common issues:**
   - Python version (must be 3.10)
   - Virtual environment not activated
   - Missing dependencies
   - Port already in use

3. **Ask team:**
   - Post in team chat with error message
   - Share screenshot
   - Mention what you tried

4. **Restart from scratch:**
   ```bash
   deactivate
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements/dev.txt
   ```

---

## ⏰ Timeline Expectations

### First 30 Minutes
- ✅ Environment setup
- ✅ Dependencies installed
- ✅ Basic tests pass

### First Hour
- ✅ Can run your role-specific first task
- ✅ Understand project structure
- ✅ Data download started (Person 2)

### First 4 Hours
- ✅ Core components created
- ✅ First model training (Person 1)
- ✅ First dashboard running (Person 3)
- ✅ Data pipeline tested (Person 2)

---

## 🚀 You're Ready!

Now proceed to your role-specific guide:
- **Person 1:** Read `PERSON1_ML_ENGINEER.md`
- **Person 2:** Read `PERSON2_DATA_ENGINEER.md`
- **Person 3:** Read `PERSON3_FULLSTACK.md`

**Good luck! You've got this! 💪**

---

## 📞 Emergency Contacts

If completely stuck:
1. Check `MASTER_PLAN.md` for overview
2. Re-read this `QUICK_START.md`
3. Check GitHub issues/documentation
4. Ask team for help

**Remember:** The goal is working software in 2 days. Ship fast, iterate later!
