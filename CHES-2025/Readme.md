# CHES Challenge 2025 - UTLA Transfer Learning Attack

This folder contains code for Unsupervised Transfer Learning Attacks (UTLA) between CHES Challenge 2025 EM traces to XMEGA power traces.

## Dataset Download

### CHES Challenge 2025 Dataset

Download the CHES Challenge 2025 dataset from Google Drive:

1. Visit the [CHES Challenge 2025](https://pace-tl.gitbook.io/ches-challenge-2025) official website for challenge details
2. Download the public dataset from [Google Drive](https://drive.google.com/drive/folders/1JGbphwZXQvN_tEhpBIbQ-q-pN9wkqKQ-?usp=sharing)
3. Extract and place `CHES_Challenge.h5` in your data directory.

### XMEGA-Power Dataset

The XMEGA-Power dataset should be organized as follows:
```
XMEGA-Power/Data/
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
| `ches-preprocessing.py` | Preprocessing script for CHES dataset |
| `Source_profiling.py` | Supervised profiling on CHES dataset |
| `UTLA_from_CHES_to_XMEGA-Power.py` | UTLA transfer: CHES → XMEGA |

## Step-by-Step Instructions

### Step 1: Preprocessing

Preprocess the CHES Challenge dataset:

```bash
python ches-preprocessing.py
```

### Step 2: Source Profiling

Train a supervised CNN model on the CHES Challenge dataset:

```bash
python Source_profiling.py 1
```

Set argument to `1` for training or `0` for inference.

### Step 3: UTLA Transfer Learning

Transfer the CHES-trained model to attack XMEGA-Power traces:

```bash
python UTLA_from_CHES_to_XMEGA-Power.py
```

## Configuration

Edit the configuration section at the top of each script to modify:

```python
# Paths
CHES_DATA_PATH = "/path/to/CHES_Challenge.h5"
XMEGA_DATA_PATH = "/path/to/XMEGA-Power/Data/device0{}"

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
| XMEGA-Power | Statistical GEEA | Strong power signals, analytical approximation works |

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
