# Unsupervised Transfer Learning Attack (UTLA)

This is a lightweight code repository for reproducing the experiments from "Attack from Shadows: Unsupervised Side-channel Transfer Learning across Devices and Modalities," to appear in AsiaCCS 2026.

The repository intentionally does not include datasets, trained models, generated NumPy result arrays, generated figures, or generated analysis folders. Download datasets from the public sources linked below and place them in the expected `Data/` subdirectories before running experiments.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The scripts use PyTorch and will use CUDA when a GPU is available.

## Repository Organization

The repository is organized by source or benchmark dataset:

1. `XMEGA-Power`: XMEGA power traces from the CDPA-SCA cross-device dataset.
2. `XMEGA-EM`: XMEGA EM traces from the CDPA-SCA cross-probe-position dataset.
3. `Sakura-AES`: Sakura AES traces from the CDPA-SCA cross-device dataset.
4. `ASCADv1-variable-key`: ASCAD variable-key metadata and placeholders.
5. `ASCADv2`: ASCADv2 profiling code and placeholders.
6. `x86-Cache`: x86 cache-timing profiling code and placeholders.
7. `CHES-2025`: CHES Challenge 2025 profiling and CHES-to-XMEGA transfer code.

Each experiment folder has a local `Readme.md` with dataset layout notes. Output files are written to `models/`, `results/`, and `figures/` subdirectories when scripts are run, but those generated artifacts are ignored by Git.

## Data Layout

For XMEGA-style datasets and the x86 cache dataset, expected local paths look like:

```text
XMEGA-Power/Data/device01/
XMEGA-EM/Data/device01/
x86-Cache/dataset.h5
```

with files such as `X_train.npy`, `X_attack.npy`, `Y_ID_train.npy`, `Y_ID_attack.npy`, `plaintexts_train.npy`, and `plaintexts_attack.npy`.

For Sakura-AES, expected device directories look like:

```text
Sakura-AES/Data/device1/
```

with files such as `X_train.npy`, `X_attack.npy`, `Y_train.npy`, `Y_attack.npy`, and `ciphertexts_attack.npy`.

For CHES Challenge 2025, place `CHES_Challenge.h5` in `CHES-2025/`, or set `CHES_DATA_PATH` and `CHES_PREPROCESSED_PATH` to external locations.

## Core Workflow

Run scripts from inside their experiment directory so relative paths resolve correctly.

Profile a source model:

```bash
python Source_Profiling.py 1 1
```

For ASCADv1 variable-key source profiling followed by ASCAD fixed-key UTLA:

```bash
cd ASCADv1-variable-key
python Source_Profiling.py 1 1
UTLA_SOURCE_CHECKPOINT=./models/pre-trained_device_ASCADv1_wGE1.pth \
  python ASCADV1_to_ASCAD_Transfer_UTLA.py 1 1 1
```

Run an in-domain or cross-domain UTLA transfer script, for example:

```bash
python XMEGA_UTLA.py 1 1 2
python UTLA_from_XMEGA-EM_to_XMEGA-EM.py 1 1 2
python UTLA_from_XMEGA-EM_to_XMEGA-Power.py 1 1 2
python UTLA_from_Sakura_to_XMEGA-Power.py 1 1 2
python x86_XMEGA_cross_UTLA.py 1 1 2
python ASCADV2_to_ASCADV1_Transfer_UTLA.py 1 1 1 1 0
```

For CHES-to-XMEGA:

```bash
cd CHES-2025
python ches-preprocessing.py
python Source_profiling.py
python UTLA_from_CHES_to_XMEGA-Power.py --mode transfer --device 2
```

## Bounded Smoke Runs

The ASCADv1 profiling and ASCADv1-to-fixed-key UTLA scripts support environment variables for quick trainability checks. These smoke runs verify data loading, one training pass, checkpoint save/load, transfer training, and result writing; they do not validate key-recovery quality.

```bash
cd ASCADv1-variable-key
export UTLA_BATCH_SIZE=20
export UTLA_TOTAL_EPOCHS=1
export UTLA_FINETUNE_EPOCHS=1
export UTLA_LOG_INTERVAL=1
export UTLA_TRAIN_NUM=40
export UTLA_SOURCE_TEST_NUM=40
export UTLA_TARGET_FINETUNE_NUM=40
export UTLA_TARGET_TEST_NUM=40
export UTLA_TRACE_NUM_MAX=40
export UTLA_GE_LOSS_TRACE_NUM_MAX=40

python Source_Profiling.py 1 0
UTLA_SOURCE_CHECKPOINT=./models/pre-trained_device_ASCADv1_wGE1.pth \
  python ASCADV1_to_ASCAD_Transfer_UTLA.py 1 1 1
```

## Public Dataset Sources

- XMEGA-Power and Sakura-AES: [CDPA-SCA Cross-Device Profiled Attack](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack)
- ASCADv1 and ASCADv2: [ANSSI-FR ASCAD](https://github.com/ANSSI-FR/ASCAD)
- x86-Cache: [Cross-Platform Cache Attack Dataset](https://github.com/Cross-Platform-Cache-Attack/x86-high-freq-cache-timing-leakage)
- CHES Challenge 2025: [official challenge page](https://pace-tl.gitbook.io/ches-challenge-2025)
