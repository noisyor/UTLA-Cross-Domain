# CHES Challenge 2025 - UTLA Transfer Learning Attack

This folder contains code for Unsupervised Transfer Learning Attacks (UTLA) between CHES Challenge 2025 EM traces to XMEGA power traces.

## Dataset Download

### CHES Challenge 2025 Dataset

Download the CHES Challenge 2025 dataset from Google Drive:

1. Visit the [CHES Challenge 2025](https://pace-tl.gitbook.io/ches-challenge-2025) official website for challenge details
2. Download the public dataset from [Google Drive](https://drive.google.com/drive/folders/1JGbphwZXQvN_tEhpBIbQ-q-pN9wkqKQ-?usp=sharing)
3. Extract and place `CHES_Challenge.h5` in your data directory.

### XMEGA Dataset

The XMEGA dataset should be organized as follows:
```
XMEGA/Data/
├── device01/
│   ├── X_train.npy
│   ├── X_attack.npy
│   ├── Y_ID_train.npy
│   ├── Y_ID_attack.npy
│   ├── plaintexts_train.npy
│   └── plaintexts_attack.npy
├── device02/
│   └── ...
└── device08/
    └── ...
```

## Files Overview

| File | Description |
|------|-------------|
| `train_ches_cnn_4layer.py` | Supervised profiling on CHES dataset |
| `utla_ches_to_xmega.py` | UTLA transfer: CHES → XMEGA |
| `utla_xmega_to_ches.py` | UTLA transfer: XMEGA → CHES |

## Step-by-Step Instructions

### Step 1: Profiling on CHES Dataset

Train a supervised CNN model on the CHES Challenge dataset:

```bash
python train_ches_cnn_4layer.py
```

**Outputs:**
- `./models/best_ntge_model.pth` - Best model checkpoint
- `./models/preprocessing_params.npy` - Preprocessing parameters
- `./figures/training_curves.png` - Training curves
- `./figures/GE_empirical_final.png` - GE curve

### Step 2: UTLA Transfer Learning

#### Option A: CHES → XMEGA Transfer

Transfer the CHES-trained model to attack XMEGA traces:

```bash
python utla_ches_to_xmega.py --mode transfer --device 2
```

**Arguments:**
- `--mode`: `pretrain` (train on CHES), `transfer` (UTLA), or `eval` (evaluate)
- `--device`: Target XMEGA device (1-8)

**Outputs:**
- `./models_utla/utla_ches_to_xmega2_phase1.pth` - After 10 epochs
- `./models_utla/utla_ches_to_xmega2_phase2.pth` - After 20 epochs
- `./figures_utla/utla_training_xmega2.png` - Training curves

#### Option B: XMEGA → CHES Transfer

Transfer the XMEGA-trained model to attack CHES traces:

```bash
python utla_xmega_to_ches.py
```

**Prerequisites:**
- Pre-trained XMEGA model: `../XMEGA-Expt/models/pre-trained_device2.pth`
- Pre-trained CHES model: `./models/best_ntge_model.pth`

**Outputs:**
- `./models_utla_xmega_to_ches/baseline.pth` - Before transfer
- `./models_utla_xmega_to_ches/phase1_epoch10.pth` - After 10 epochs
- `./models_utla_xmega_to_ches/phase2_epoch20.pth` - After 20 epochs
- `./figures_utla_xmega_to_ches/training_curves.png` - Training curves

## Configuration

Edit the configuration section at the top of each script to modify:

```python
# Paths
CHES_DATA_PATH = "/path/to/CHES_Challenge.h5"
XMEGA_DATA_PATH = "/path/to/XMEGA/Data/device0{}"

# Training parameters
BATCH_SIZE = 256
TRANSFER_EPOCHS = 20
ENCODER_LR = 1e-4

# MMD loss weights (from paper)
LAMBDA1 = 2.0    # Encoder output MMD
LAMBDA2 = 0.05   # Penultimate layer MMD
```

## NTGE Computation Methods

| Dataset | Method | Reason |
|---------|--------|--------|
| CHES | Empirical Monte Carlo | Weak EM signals, statistical methods fail |
| XMEGA | Statistical GEEA | Strong power signals, analytical approximation works |

## Requirements

```
torch>=1.9.0
numpy
h5py
scipy
scikit-learn
matplotlib
tqdm
```

## References

- [CHES Challenge 2025](https://pace-tl.gitbook.io/ches-challenge-2025)
- [CHES Challenge 2025 Dataset](https://drive.google.com/drive/folders/1JGbphwZXQvN_tEhpBIbQ-q-pN9wkqKQ-)
