# 📊 Dataset Recommendations for Multimodal Anomaly Detection

**Analysis of 10 datasets for your 2-day implementation**

---

## 🎯 RECOMMENDED DATASETS (Use These!)

### ✅ **PRIMARY DATASETS** (Must Use - Perfect for Multimodal)

#### 1. **NASA C-MAPSS Turbofan Engine** ⭐⭐⭐⭐⭐
**Link:** https://www.kaggle.com/datasets/behrad3d/nasa-cmaps/data

**Why Perfect:**
- ✅ **Time-Series Focus:** 21 sensors × multiple cycles
- ✅ **Well-Structured:** Clean, standardized format
- ✅ **Proven Baseline:** Widely used in research (easy to validate)
- ✅ **Labeled Data:** Run-to-failure trajectories with RUL labels
- ✅ **Multiple Scenarios:** 4 different operating conditions

**Modalities:**
- Time-series sensor data (temperature, pressure, vibration, RPM, etc.)

**Size:** ~10 MB (manageable)

**Use For:**
- Time-Series VAE training
- Primary anomaly detection evaluation
- Root cause analysis demonstration

**Download Command:**
```bash
kaggle datasets download -d behrad3d/nasa-cmaps
unzip nasa-cmaps.zip -d data/raw/nasa_turbofan/
```

---

#### 2. **MIMII Sound Dataset** ⭐⭐⭐⭐⭐
**Link:** https://zenodo.org/records/3384388

**Why Perfect:**
- ✅ **Audio Modality:** Industrial machine sounds (fan, pump, valve, slider)
- ✅ **Anomaly Detection:** Normal + anomalous recordings
- ✅ **Real Industrial Data:** From actual factory equipment
- ✅ **Multiple SNR Levels:** -6dB, 0dB, 6dB

**Modalities:**
- Audio (WAV files)
- Can convert to spectrograms (time-frequency images)

**Size:** ~3-4 GB per machine type

**Use For:**
- Audio anomaly detection (convert to spectrograms)
- Alternative time-series modality
- Multimodal fusion with other sensors

**Download:**
```bash
wget https://zenodo.org/record/3384388/files/6_dB_fan.zip
unzip 6_dB_fan.zip -d data/raw/mimii/
```

**Pro Tip:** Focus on ONE machine type (fan) to save time!

---

#### 3. **Thermal Image Dataset** ⭐⭐⭐⭐
**Link:** https://www.kaggle.com/datasets/animeshmahajan/thermal-image-dataset

**Why Good:**
- ✅ **Vision Modality:** 7,412 thermal images
- ✅ **Pre-labeled:** Manually annotated
- ✅ **Practical:** Thermal imaging used in real predictive maintenance
- ✅ **Deep Learning Ready:** Good for Vision VAE

**Modalities:**
- Thermal infrared images

**Size:** ~500 MB (estimated)

**Use For:**
- Vision VAE training
- Thermal anomaly detection
- Multimodal fusion (thermal + sensor)

**Download:**
```bash
kaggle datasets download -d animeshmahajan/thermal-image-dataset
unzip thermal-image-dataset.zip -d data/raw/thermal/
```

**Note:** Originally for person detection, but thermal patterns are excellent for anomaly detection training!

---

#### 4. **Bosch CNC Machining Dataset** ⭐⭐⭐⭐
**Link:** https://github.com/boschresearch/CNC_Machining

**Why Excellent:**
- ✅ **Real Industrial Data:** From actual CNC machines
- ✅ **Vibration Data:** 3-axis accelerometer (X, Y, Z)
- ✅ **Labeled Anomalies:** "good" vs "bad" labels
- ✅ **Multiple Machines:** 3 machines, 15 processes
- ✅ **High Frequency:** 2 kHz sampling rate

**Modalities:**
- Time-series vibration data

**Size:** ~100-200 MB

**Use For:**
- High-frequency time-series VAE
- Industrial vibration anomaly detection
- Validation of your model on different domain

**Download:**
```bash
git clone https://github.com/boschresearch/CNC_Machining.git
mv CNC_Machining/data data/raw/cnc_machining/
```

---

### ⚠️ **SECONDARY DATASETS** (Use if Time Permits)

#### 5. **MetroPT-3 Dataset** ⭐⭐⭐
**Link:** https://www.kaggle.com/datasets/joebeachcapital/metropt-3-dataset

**Why Useful:**
- Real metro train compressor data
- 15 sensors (analogue + digital)
- Good for RUL prediction

**Modalities:**
- Time-series (pressure, temperature, current)

**Use For:** Additional validation dataset

**Download:**
```bash
kaggle datasets download -d joebeachcapital/metropt-3-dataset
```

---

#### 6. **Smart Manufacturing Multi-Agent** ⭐⭐⭐
**Link:** https://www.kaggle.com/datasets/ziya07/smart-manufacturing-multi-agent-control-dataset

**Why Interesting:**
- Multi-agent control data
- Includes cyber-security anomalies
- System efficiency metrics

**Modalities:**
- Time-series control signals
- Anomaly labels

**Use For:** Cyber-physical anomaly detection (bonus feature)

---

### ❌ **NOT RECOMMENDED** (Skip These)

#### 7. **UNSW-NB15**
**Link:** https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

**Why Skip:**
- ❌ Network intrusion detection (not industrial)
- ❌ Not aligned with predictive maintenance focus
- ❌ Different domain entirely

---

#### 8. **DDoS Traffic Dataset**
**Link:** https://www.kaggle.com/datasets/kalireadhat/realtime-ddos-traffic-dataset

**Why Skip:**
- ❌ Cybersecurity focus (not industrial equipment)
- ❌ Wrong modality (network packets)
- ❌ Out of scope

---

#### 9. **Oil & Gas Pipeline**
**Link:** https://www.kaggle.com/datasets/muhammadwaqas023/predictive-maintenance-oil-and-gas-pipeline-data

**Why Skip:**
- ❌ Only 1,000 synthetic samples (too small)
- ❌ Binary classification (not rich enough)
- ❌ Synthetic data (less impressive)

---

#### 10. **Freederia Aviation Anomaly**
**Link:** https://www.kaggle.com/datasets/freederiaresearch/automated-anomaly-detection-and-predictive-mitigat

**Why Skip:**
- ❌ Very small (8 KB)
- ❌ Limited information available
- ❌ Better alternatives exist

---

## 🎯 FINAL RECOMMENDATION: Your Multimodal Stack

### **For 2-Day Implementation, Use These 3:**

```
┌─────────────────────────────────────────────────────┐
│  MODALITY 1: Time-Series Sensors                    │
│  Dataset: NASA C-MAPSS Turbofan                     │
│  Size: ~10 MB                                        │
│  Download Time: 2-3 minutes                         │
│  Training Time: 1-2 hours                           │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  MODALITY 2: Audio → Spectrograms                   │
│  Dataset: MIMII Sound (Fan only)                    │
│  Size: ~3 GB                                         │
│  Download Time: 30-60 minutes                       │
│  Training Time: 2-3 hours                           │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  MODALITY 3: Vision (Thermal Images)                │
│  Dataset: Thermal Image Dataset                     │
│  Size: ~500 MB                                       │
│  Download Time: 10-15 minutes                       │
│  Training Time: 3-4 hours                           │
└─────────────────────────────────────────────────────┘

Optional 4th: Bosch CNC Machining (if time permits)
```

---

## 📥 Quick Download Script

Create: `scripts/download_recommended_datasets.sh`

```bash
#!/bin/bash

echo "=============================================="
echo "Downloading Recommended Datasets for 2-Day Project"
echo "=============================================="

# Create directories
mkdir -p data/raw/nasa_turbofan
mkdir -p data/raw/mimii
mkdir -p data/raw/thermal
mkdir -p data/raw/cnc_machining

# 1. NASA C-MAPSS Turbofan (PRIORITY 1)
echo ""
echo "1. Downloading NASA C-MAPSS Turbofan Dataset..."
cd data/raw/nasa_turbofan
kaggle datasets download -d behrad3d/nasa-cmaps
unzip -o nasa-cmaps.zip
rm nasa-cmaps.zip
cd ../../..
echo "✅ NASA Turbofan downloaded"

# 2. MIMII Sound Dataset - Fan only (PRIORITY 2)
echo ""
echo "2. Downloading MIMII Sound Dataset (Fan - 6dB SNR)..."
cd data/raw/mimii
wget https://zenodo.org/record/3384388/files/6_dB_fan.zip
unzip -o 6_dB_fan.zip
rm 6_dB_fan.zip
cd ../../..
echo "✅ MIMII Sound (Fan) downloaded"

# 3. Thermal Images (PRIORITY 3)
echo ""
echo "3. Downloading Thermal Image Dataset..."
cd data/raw/thermal
kaggle datasets download -d animeshmahajan/thermal-image-dataset
unzip -o thermal-image-dataset.zip
rm thermal-image-dataset.zip
cd ../../..
echo "✅ Thermal Images downloaded"

# 4. Bosch CNC (OPTIONAL)
echo ""
echo "4. Downloading Bosch CNC Machining Dataset (Optional)..."
git clone https://github.com/boschresearch/CNC_Machining.git temp_cnc
mv temp_cnc/data/* data/raw/cnc_machining/
rm -rf temp_cnc
echo "✅ Bosch CNC downloaded"

echo ""
echo "=============================================="
echo "All recommended datasets downloaded!"
echo "=============================================="
echo ""
echo "Dataset locations:"
echo "  - NASA Turbofan: data/raw/nasa_turbofan/"
echo "  - MIMII Sound: data/raw/mimii/"
echo "  - Thermal Images: data/raw/thermal/"
echo "  - CNC Machining: data/raw/cnc_machining/"
echo ""
echo "Total size: ~4-5 GB"
echo "Ready for training!"
```

**Make executable:**
```bash
chmod +x scripts/download_recommended_datasets.sh
```

---

## 🔄 Multimodal Fusion Strategy

### **Approach 1: Sequential (Recommended for 2 Days)**

```
Day 1 Morning:  Train Time-Series VAE (NASA Turbofan)
Day 1 Afternoon: Train Audio VAE (MIMII → Spectrograms)
Day 2 Morning:  Train Vision VAE (Thermal Images)
Day 2 Afternoon: Multimodal Fusion
```

### **Approach 2: If Short on Time**

Use just **NASA Turbofan** + **Thermal Images**:
- Faster download (~500 MB total)
- Clear multimodal distinction (sensors + vision)
- Still demonstrates full pipeline

---

## 📊 Dataset Comparison Table

| Dataset | Modality | Size | Download Time | Use Case | Priority |
|---------|----------|------|---------------|----------|----------|
| NASA C-MAPSS | Time-Series | 10 MB | 2 min | Primary sensor data | ⭐⭐⭐⭐⭐ |
| MIMII Sound | Audio/Spectrogram | 3 GB | 30-60 min | Audio anomalies | ⭐⭐⭐⭐⭐ |
| Thermal Images | Vision | 500 MB | 10 min | Visual anomalies | ⭐⭐⭐⭐ |
| Bosch CNC | Vibration | 200 MB | 5 min | Validation | ⭐⭐⭐⭐ |
| MetroPT-3 | Time-Series | 50 MB | 5 min | Additional validation | ⭐⭐⭐ |
| Smart Mfg | Multi-Agent | 3.4 MB | 1 min | Bonus feature | ⭐⭐⭐ |

---

## ⏰ Time Allocation

### **If You Have Full 2 Days:**
- ✅ NASA Turbofan (must have)
- ✅ MIMII Sound (must have)
- ✅ Thermal Images (must have)
- ✅ Bosch CNC (bonus validation)

### **If Running Behind Schedule:**
- ✅ NASA Turbofan (must have)
- ✅ Thermal Images (must have)
- ⏭️ Skip MIMII (audio)
- ⏭️ Skip CNC

**Minimum Viable:** NASA + Thermal = Still multimodal!

---

## 🚀 Pre-Download Before Day 1!

**STRONGLY RECOMMENDED:**

```bash
# Run this the night before you start
nohup ./scripts/download_recommended_datasets.sh > download.log 2>&1 &

# Check progress
tail -f download.log
```

This way datasets are ready when you start!

---

## 📝 Dataset Credits

- **NASA C-MAPSS:** NASA Ames Prognostics Center
- **MIMII:** Hitachi, Ltd. & ToyotaTechnological Institute
- **Thermal Images:** Various contributors on Kaggle
- **Bosch CNC:** Bosch Research

---

## ✅ Validation Checklist

After download, verify:

```bash
# Check NASA Turbofan
ls data/raw/nasa_turbofan/train_FD001.txt  # Should exist

# Check MIMII
ls data/raw/mimii/fan/train/normal/  # Should have .wav files

# Check Thermal
ls data/raw/thermal/*.jpg  # Should have images

# Check sizes
du -sh data/raw/*
```

**Expected:**
```
10M     data/raw/nasa_turbofan
3.0G    data/raw/mimii
500M    data/raw/thermal
200M    data/raw/cnc_machining (optional)
```

---

## 🎯 Interview Talking Points

**"We used a diverse multimodal dataset stack:"**

1. **"NASA C-MAPSS Turbofan for time-series sensor anomalies"**
   - 21 sensors, run-to-failure data
   - Industry-standard benchmark

2. **"MIMII industrial sound dataset for audio-based detection"**
   - Converted audio to spectrograms
   - Real factory equipment sounds

3. **"Thermal imaging dataset for visual anomalies"**
   - Thermal cameras used in real predictive maintenance
   - 7,400+ labeled images

4. **"Product-of-Experts fusion to combine all modalities"**
   - Precision-weighted multimodal learning
   - Higher accuracy than single-modality

**This demonstrates real-world industrial AI system design!**

---

## 🔗 Quick Links

**Setup Kaggle:**
```bash
pip install kaggle
mkdir ~/.kaggle
# Download kaggle.json from https://www.kaggle.com/settings
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**All Download Commands:**
```bash
# NASA Turbofan
kaggle datasets download -d behrad3d/nasa-cmaps

# Thermal Images
kaggle datasets download -d animeshmahajan/thermal-image-dataset

# MIMII Sound
wget https://zenodo.org/record/3384388/files/6_dB_fan.zip

# Bosch CNC
git clone https://github.com/boschresearch/CNC_Machining.git

# MetroPT-3
kaggle datasets download -d joebeachcapital/metropt-3-dataset
```

---

**FINAL VERDICT: Use NASA Turbofan + MIMII + Thermal Images for complete multimodal coverage! 🎯**
