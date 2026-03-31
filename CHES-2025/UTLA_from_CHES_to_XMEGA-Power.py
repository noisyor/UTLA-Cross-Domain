#!/usr/bin/env python
# coding: utf-8
"""
UTLA (Unsupervised Transfer Learning Attack) from CHES to XMEGA.

Source domain: CHES Challenge 2025 (EM traces)
Target domain: XMEGA power traces

The classifier is pre-trained on CHES and kept frozen.
The encoder is adapted using adversarial domain adaptation + MMD loss.

Usage:
    python UTLA_from_CHES_to_XMEGA-Power.py --mode pretrain   # Pre-train on CHES
    python UTLA_from_CHES_to_XMEGA-Power.py --mode transfer   # Transfer to XMEGA
    python UTLA_from_CHES_to_XMEGA-Power.py --mode eval       # Evaluate on XMEGA
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

# Paths
CHES_DATA_PATH = os.environ.get("CHES_DATA_PATH", "./CHES_Challenge.h5")
CHES_PREPROCESSED_PATH = os.environ.get("CHES_PREPROCESSED_PATH", "./CHES_Challenge_preprocessed.h5")
USE_PREPROCESSED = True  # Use preprocessed CHES traces (faster loading)

XMEGA_DATA_PATH = os.environ.get("XMEGA_DATA_PATH", "../XMEGA-Power/Data/device0{}")
CHES_PRETRAINED_PATH = "./models/best_ntge_model.pth"  # 4-layer CHES model
OUTPUT_DIR = "./models_utla"
FIGURES_DIR = "./figures_utla"
RESULTS_DIR = "./results_utla"

# Architecture - use 4-layer to match XMEGA exactly
USE_4LAYER_ARCH = True  # True: 4-layer (matches XMEGA), False: 3-layer (original CHES)

# CHES dataset parameters
CHES_TRAIN_NUM = 400000
CHES_VALID_NUM = 50000
CHES_TRACE_OFFSET = 1200
CHES_TRACE_LENGTH = 500  # Must give flatten_size=192

# Preprocessing parameters (used when USE_PREPROCESSED=False)
USE_FILTERING = True      # Apply low-pass filter before standardization
FILTER_ORDER = 5          # Butterworth filter order
FILTER_CUTOFF = 0.3       # Normalized cutoff frequency (0.3 = 30% of Nyquist)

# XMEGA dataset parameters
XMEGA_TRAIN_NUM = 40000
XMEGA_VALID_NUM = 5000
XMEGA_TEST_NUM = 5000
XMEGA_TRACE_OFFSET = 0
XMEGA_TRACE_LENGTH = 500  # Same as CHES for architecture compatibility
XMEGA_TARGET_DEVICE = 2  # Target XMEGA device (1-8)

# Training parameters (Section 6.1 and Supplementary H)
BATCH_SIZE = 256
PRETRAIN_EPOCHS = 50
TRANSFER_EPOCHS = 20  # Reduced for diagnostic (paper: 20-50 epochs)
PRETRAIN_LR = 1e-3
ENCODER_LR = 1e-4      # Paper: "1e-5 and 1e-3" (Supplementary H)
DISCRIMINATOR_LR = 1e-4  # Paper: "smaller learning rate for discriminator"

# MMD loss weights - FROM PAPER Section 6.1
# "We apply MMD loss at both the encoder's output (λ1 = 2) and its penultimate layer (λ2 = 0.05)"
LAMBDA1 = 2.0    # Weight for encoder output (features_4) MMD
LAMBDA2 = 0.05   # Weight for penultimate layer (features_3) MMD

# Encoder initialization options:
# 'random'  - Random init (paper recommends for heterogeneous transfer)
# 'source'  - Copy from source (CHES) model
# 'xmega'   - Use supervised XMEGA model (for diagnostic: start from good solution)
ENCODER_INIT = 'xmega'  # Options: 'random', 'source', 'xmega'

# Path to supervised XMEGA model (used when ENCODER_INIT='xmega')
XMEGA_SUPERVISED_PATH = os.environ.get(
    "XMEGA_SUPERVISED_PATH",
    "../XMEGA-Power/models/pre-trained_device2.pth",
)

USE_ADVERSARIAL = True  # Use full ADDA+MMD (Eq. 15)

# Attack parameters
TRACE_NUM_MAX = 5000
CLASS_NUM = 256
N_EXPERIMENTS = 100

# Early stopping
PATIENCE = 20  # High to avoid early stopping in diagnostic runs

# Device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# AES S-box
Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])


# ============================================================================
# Dataset Classes
# ============================================================================

class CHESDataset(Dataset):
    """Dataset class for CHES Challenge 2025."""
    
    def __init__(self, traces, labels, plaintexts, trace_offset, trace_length):
        self.traces = traces.astype(np.float32)
        self.labels = labels
        self.plaintexts = plaintexts
        self.trace_offset = trace_offset
        self.trace_length = trace_length
    
    def __getitem__(self, i):
        trace = self.traces[i, self.trace_offset:self.trace_offset + self.trace_length]
        trace = torch.tensor(trace.reshape(1, -1), dtype=torch.float32)
        label = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return trace, label, i
    
    def __len__(self):
        return len(self.traces)


class XMEGADataset(Dataset):
    """Dataset class for XMEGA power traces."""
    
    def __init__(self, traces, labels, plaintexts, trace_offset, trace_length):
        self.traces = traces.astype(np.float32)
        self.labels = labels
        self.plaintexts = plaintexts
        self.trace_offset = trace_offset
        self.trace_length = trace_length
    
    def __getitem__(self, i):
        trace = self.traces[i, self.trace_offset:self.trace_offset + self.trace_length]
        trace = torch.tensor(trace.reshape(1, -1), dtype=torch.float32)
        label = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return trace, label, i
    
    def __len__(self):
        return len(self.traces)


# ============================================================================
# Model Architecture
# ============================================================================

class UTLA_Net_3Layer(nn.Module):
    """
    UTLA CNN architecture (3-layer encoder).
    Used for CHES model compatibility.
    """
    
    def __init__(self, trace_length, num_classes=256):
        super(UTLA_Net_3Layer, self).__init__()
        
        self.trace_length = trace_length
        
        # Encoder (features) - 3 layers
        self.features_1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=1),
            nn.SELU(),
            nn.BatchNorm1d(8),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        
        self.features_2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=11),
            nn.SELU(),
            nn.BatchNorm1d(16),
            nn.AvgPool1d(kernel_size=11, stride=11)
        )
        
        self.features_3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=2),
            nn.SELU(),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(kernel_size=3, stride=3),
            nn.Flatten()
        )
        
        # Calculate flatten size
        self._calculate_flatten_size()
        
        # Classifier - 2 layers
        self.classifier_1 = nn.Sequential(
            nn.Linear(self.flatten_size, 2),
            nn.SELU(),
        )
        
        self.final_classifier = nn.Sequential(
            nn.Linear(2, num_classes)
        )
    
    def _calculate_flatten_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.trace_length)
            x = self.features_1(dummy)
            x = self.features_2(x)
            x = self.features_3(x)
            self.flatten_size = x.shape[1]
        print(f"Flatten size: {self.flatten_size}")
    
    def forward(self, x):
        x = self.features_1(x)
        feat_m1 = self.features_2(x)  # Penultimate encoder layer
        feat = self.features_3(feat_m1)  # Final encoder output
        feat = feat.view(feat.size(0), -1)
        feat_clf = self.classifier_1(feat)
        output = self.final_classifier(feat_clf)
        
        return output, feat, feat_m1.view(feat_m1.size(0), -1)
    
    def encode(self, x):
        """Get encoder output only."""
        x = self.features_1(x)
        x = self.features_2(x)
        x = self.features_3(x)
        return x.view(x.size(0), -1)


class UTLA_Net_4Layer(nn.Module):
    """
    UTLA CNN architecture (4-layer encoder).
    Matches the XMEGA supervised model architecture exactly.
    
    Architecture from XMEGA direct transfer code:
    - features_1: Conv1d(1, 8, k=1), Pool(2,2)
    - features_2: Conv1d(8, 16, k=9), Pool(9,9)
    - features_3: Conv1d(16, 32, k=2), Pool(3,3)
    - features_4: Conv1d(32, 64, k=2), Pool(2,2), Flatten
    - classifier_1: Linear(192, 2)
    - final_classifier: Linear(2, 256)
    """
    
    def __init__(self, trace_length, num_classes=256):
        super(UTLA_Net_4Layer, self).__init__()
        
        self.trace_length = trace_length
        
        # Encoder (features) - 4 layers (matches XMEGA)
        self.features_1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=1),
            nn.SELU(),
            nn.BatchNorm1d(8),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        
        self.features_2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=9),
            nn.SELU(),
            nn.BatchNorm1d(16),
            nn.AvgPool1d(kernel_size=9, stride=9)
        )
        
        self.features_3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=2),
            nn.SELU(),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(kernel_size=3, stride=3)
        )
        
        self.features_4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=2),
            nn.SELU(),
            nn.BatchNorm1d(64),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        # Calculate flatten size
        self._calculate_flatten_size()
        
        # Classifier - 2 layers
        self.classifier_1 = nn.Sequential(
            nn.Linear(self.flatten_size, 2),
            nn.SELU(),
        )
        
        self.final_classifier = nn.Sequential(
            nn.Linear(2, num_classes)
        )
    
    def _calculate_flatten_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.trace_length)
            x = self.features_1(dummy)
            x = self.features_2(x)
            x = self.features_3(x)
            x = self.features_4(x)
            self.flatten_size = x.shape[1]
        print(f"Flatten size: {self.flatten_size}")
    
    def forward(self, x):
        x = self.features_1(x)
        x = self.features_2(x)
        feat_m1 = self.features_3(x)  # Penultimate encoder layer (for λ2 MMD)
        feat = self.features_4(feat_m1)  # Final encoder output (for λ1 MMD)
        feat = feat.view(feat.size(0), -1)
        feat_clf = self.classifier_1(feat)
        output = self.final_classifier(feat_clf)
        
        return output, feat, feat_m1.view(feat_m1.size(0), -1)
    
    def encode(self, x):
        """Get encoder output only."""
        x = self.features_1(x)
        x = self.features_2(x)
        x = self.features_3(x)
        x = self.features_4(x)
        return x.view(x.size(0), -1)


# Alias for backward compatibility
UTLA_Net = UTLA_Net_3Layer


class Discriminator(nn.Module):
    """Domain discriminator for adversarial training."""
    
    def __init__(self, input_size=192):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.SELU(),
            nn.Linear(64, 2)  # 2 classes: source vs target
        )
    
    def forward(self, x):
        return self.discriminator(x)


# ============================================================================
# MMD Loss Functions
# ============================================================================

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Compute Gaussian kernel matrix."""
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    
    L2_distance = ((total0 - total1) ** 2).sum(2)
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    
    return sum(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Compute MMD loss with RBF kernel."""
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    
    loss = torch.mean(XX + YY - XY - YX)
    return loss


# ============================================================================
# Data Loading
# ============================================================================

def load_ches_data(filepath, byte=0):
    """Load CHES Challenge dataset (raw)."""
    print(f"Loading CHES dataset from {filepath}...")
    
    with h5py.File(filepath, "r") as f:
        X_profiling = np.array(f['Profiling_traces/traces'])
        P_profiling = np.array(f['Profiling_traces/metadata'][:]['plaintext'][:, byte])
        K_profiling = np.array(f['Profiling_traces/metadata'][:]['key'][:, byte])
        
        if byte == 0:
            Y_profiling = np.array(f['Profiling_traces/metadata'][:]['labels'])
        else:
            Y_profiling = np.array([Sbox[p ^ k] for p, k in zip(P_profiling, K_profiling)])
        
        X_attack = np.array(f['Attack_traces/traces'])
        P_attack = np.array(f['Attack_traces/metadata'][:]['plaintext'][:, byte])
        K_attack = np.array(f['Attack_traces/metadata'][:]['key'][:, byte])
        
        if byte == 0:
            Y_attack = np.array(f['Attack_traces/metadata'][:]['labels'])
        else:
            Y_attack = np.array([Sbox[p ^ k] for p, k in zip(P_attack, K_attack)])
    
    correct_key = K_attack[0]
    print(f"  Profiling: {X_profiling.shape}, Attack: {X_attack.shape}")
    print(f"  Correct key: {correct_key}")
    
    return {
        'X_train': X_profiling,
        'Y_train': Y_profiling,
        'P_train': P_profiling,
        'X_attack': X_attack,
        'Y_attack': Y_attack,
        'P_attack': P_attack,
        'correct_key': correct_key
    }


def load_ches_preprocessed(filepath):
    """
    Load preprocessed CHES dataset (already windowed and standardized).
    
    This is much faster than loading raw data and preprocessing on-the-fly.
    Use preprocess_ches.py to generate the preprocessed file.
    """
    print(f"Loading PREPROCESSED CHES dataset from {filepath}...")
    
    with h5py.File(filepath, "r") as f:
        # Check metadata
        trace_offset = f.attrs.get('trace_offset', 'unknown')
        trace_length = f.attrs.get('trace_length', 'unknown')
        preprocessing = f.attrs.get('preprocessing', 'unknown')
        print(f"  Preprocessing: {preprocessing}")
        print(f"  Window: offset={trace_offset}, length={trace_length}")
        
        # Load training data
        X_train = np.array(f['train/traces'])
        Y_train = np.array(f['train/labels'])
        P_train = np.array(f['train/plaintext'])
        
        # Load attack data
        X_attack = np.array(f['attack/traces'])
        P_attack = np.array(f['attack/plaintext'])
        correct_key = int(f['attack/correct_key'][()])
    
    print(f"  Train: {X_train.shape}, Attack: {X_attack.shape}")
    print(f"  Correct key: {correct_key}")
    
    # Note: preprocessed data doesn't have Y_attack (not needed for attack)
    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'P_train': P_train,
        'X_attack': X_attack,
        'Y_attack': None,  # Not stored in preprocessed file
        'P_attack': P_attack,
        'correct_key': correct_key,
        'preprocessed': True  # Flag to skip preprocessing
    }


def load_xmega_data(base_path, device_id):
    """Load XMEGA dataset for a specific device."""
    file_path = base_path.format(device_id) + '/'
    print(f"Loading XMEGA device {device_id} from {file_path}...")
    
    X_train = np.load(file_path + 'X_train.npy')
    X_attack = np.load(file_path + 'X_attack.npy')
    Y_train = np.load(file_path + 'Y_ID_train.npy')
    Y_attack = np.load(file_path + 'Y_ID_attack.npy')
    P_train = np.load(file_path + 'plaintexts_train.npy')
    P_attack = np.load(file_path + 'plaintexts_attack.npy')
    
    correct_key = device_id  # XMEGA keys are 0x01-0x08
    
    print(f"  Train: {X_train.shape}, Attack: {X_attack.shape}")
    print(f"  Correct key: {correct_key}")
    
    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'P_train': P_train,
        'X_attack': X_attack,
        'Y_attack': Y_attack,
        'P_attack': P_attack,
        'correct_key': correct_key
    }


# ============================================================================
# Preprocessing
# ============================================================================

def butter_lowpass_filter(data, cutoff=0.3, order=5):
    """
    Apply Butterworth low-pass filter to remove high-frequency noise.
    
    Args:
        data: numpy array of shape (N, T) - N traces, T time samples
        cutoff: Normalized cutoff frequency (0 < cutoff < 1, where 1 = Nyquist)
        order: Filter order
    
    Returns:
        Filtered traces of same shape
    """
    from scipy.signal import butter, filtfilt
    b, a = butter(order, cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=1)


def preprocess_per_trace(X, use_filter=False, cutoff=0.3, order=5):
    """
    Preprocessing pipeline for traces.
    
    Steps:
    1. Low-pass filter (optional) - remove high-frequency noise
    2. Per-trace standardization (zero mean, unit variance)
    
    Args:
        X: numpy array of shape (N, T)
        use_filter: Whether to apply low-pass filter
        cutoff: Filter cutoff frequency (normalized)
        order: Filter order
    
    Returns:
        Preprocessed traces
    """
    X = X.astype(np.float32)
    
    # Step 1: Low-pass filter (optional)
    if use_filter:
        print("Applying Butterworth low-pass filter...")
        X = butter_lowpass_filter(X, cutoff, order)
    
    # Step 2: Per-trace standardization
    print("Applying per-trace standardization...")
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std == 0] = 1
    X = (X - mean) / std
    
    return X.astype(np.float32)


def preprocess_global(X_train, X_test=None):
    """Global standardization using training statistics."""
    print("Applying global standardization...")
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1
    
    X_train_norm = ((X_train - mean) / std).astype(np.float32)
    
    if X_test is not None:
        X_test_norm = ((X_test - mean) / std).astype(np.float32)
        return X_train_norm, X_test_norm, mean, std
    
    return X_train_norm, mean, std


# ============================================================================
# Empirical NTGE Computation (for CHES - weak signals)
# ============================================================================

def compute_empirical_ntge(preds, plaintexts, real_key, n_experiments=100):
    """Compute NTGE using empirical Monte Carlo method."""
    n_traces = min(TRACE_NUM_MAX, len(preds))
    preds = preds[:n_traces].astype(np.float64)
    plaintexts = plaintexts[:n_traces]
    
    eps = 1e-40
    preds = np.clip(preds, eps, 1.0)
    log_preds = np.log(preds)
    
    pt_range = np.arange(256, dtype=np.uint8)
    key_range = np.arange(256, dtype=np.uint8)
    label_table = Sbox[pt_range[:, None] ^ key_range[None, :]]
    
    guess_labels = label_table[plaintexts, :]
    trace_idx = np.arange(n_traces)[:, None]
    log_probs_per_key = log_preds[trace_idx, guess_labels]
    
    ntge_list = []
    ge_curves = []
    
    for exp in range(n_experiments):
        perm = np.random.permutation(n_traces)
        log_probs_shuffled = log_probs_per_key[perm, :]
        cumsum = np.cumsum(log_probs_shuffled, axis=0)
        
        correct_cumsum = cumsum[:, real_key]
        keys_beating_correct = np.sum(cumsum > correct_cumsum[:, None], axis=1)
        ge_curve = 1 + keys_beating_correct
        ge_curves.append(ge_curve)
        
        indices = np.where(ge_curve == 1)[0]
        if len(indices) > 0:
            ntge_list.append(indices[0] + 1)
    
    mean_ge_curve = np.mean(ge_curves, axis=0)
    
    if len(ntge_list) > 0:
        ntge_mean = int(np.mean(ntge_list))
        ntge_median = int(np.median(ntge_list))
        success_rate = len(ntge_list) / n_experiments * 100
    else:
        ntge_mean = float('inf')
        ntge_median = float('inf')
        success_rate = 0
    
    return ntge_mean, ntge_median, success_rate, mean_ge_curve


# ============================================================================
# Statistical GEEA (for XMEGA - strong signals)
# ============================================================================

def compute_statistical_ge(preds, plaintexts, real_key):
    """
    Compute GE and NTGE using statistical GEEA method.
    
    This uses the analytical Gaussian approximation from:
    "A Fast and Accurate Guessing Entropy Estimation Algorithm" (Zhang et al., TCHES 2020)
    
    Works well for strong signals (like XMEGA power traces).
    For weak signals (like CHES EM traces), use compute_empirical_ntge().
    
    Args:
        preds: numpy array (N, 256) - model output probabilities
        plaintexts: numpy array (N,) - plaintext bytes
        real_key: int - correct key byte
    
    Returns:
        ntge: estimated NTGE (first n where GE=1)
        ge_dict: dict of GE at sample points
        ge_curve: GE at each trace count (for plotting)
    """
    n_traces = min(TRACE_NUM_MAX, len(preds))
    score_trace_num = min(10000, n_traces)  # Use subset for mean/var estimation
    
    preds = preds[:n_traces].astype(np.float64)
    plaintexts = plaintexts[:n_traces]
    
    # Clip and log
    eps = 1e-40
    preds = np.clip(preds, eps, 1.0)
    log_preds = np.log(preds)
    
    # Compute score differences: log P(label_k) - log P(label_correct)
    # For each trace and each wrong key guess
    mean_est = np.zeros(256)
    var_est = np.zeros(256)
    
    # Vectorized computation
    pt_range = np.arange(256, dtype=np.uint8)
    key_range = np.arange(256, dtype=np.uint8)
    label_table = Sbox[pt_range[:, None] ^ key_range[None, :]]
    
    # Get labels for all key guesses
    guess_labels = label_table[plaintexts[:score_trace_num], :]  # (n, 256)
    trace_idx = np.arange(score_trace_num)[:, None]
    
    # Log-probs for each key guess
    log_probs = log_preds[:score_trace_num][trace_idx, guess_labels]  # (n, 256)
    
    # Correct key log-probs
    correct_labels = Sbox[plaintexts[:score_trace_num] ^ real_key]
    correct_log_probs = log_preds[:score_trace_num][np.arange(score_trace_num), correct_labels]  # (n,)
    
    # Score difference: wrong key - correct key
    score_diff = log_probs - correct_log_probs[:, None]  # (n, 256)
    
    # Mean and variance for each key
    mean_est = np.mean(score_diff, axis=0)
    var_est = np.var(score_diff, axis=0)
    
    # Compute GE at sample points using GEEA formula
    # GE(q) = 1 + sum_{k != correct} Phi(sqrt(q) * mu_k / sigma_k)
    sample_points = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    sample_points = [n for n in sample_points if n <= n_traces]
    
    ge_dict = {}
    ge_curve = np.zeros(n_traces)
    
    for n in range(1, n_traces + 1):
        ge = 1.0
        for k in range(256):
            if k != real_key:
                if var_est[k] > 1e-10:
                    z_score = np.sqrt(n) * mean_est[k] / np.sqrt(var_est[k])
                    ge += norm.cdf(z_score)
                else:
                    # If variance is 0, check sign of mean
                    if mean_est[k] > 0:
                        ge += 1.0
                    elif mean_est[k] == 0:
                        ge += 0.5
        ge_curve[n-1] = ge
        if n in sample_points:
            ge_dict[n] = ge
    
    # Find NTGE (first n where GE <= 1.5, rounding to 1)
    ntge = n_traces  # Default: didn't reach GE=1
    for n in range(1, n_traces + 1):
        if ge_curve[n-1] <= 1.5:  # GE effectively = 1
            ntge = n
            break
    
    # Print diagnostics
    n_positive = np.sum((mean_est > 0) & (np.arange(256) != real_key))
    print(f"  Statistical GE diagnostics:")
    print(f"    Wrong keys with positive mean: {n_positive}/255")
    print(f"    GE at sample points: ", end="")
    for n in sample_points[:6]:
        print(f"GE[{n}]={ge_dict[n]:.1f} ", end="")
    print()
    
    return ntge, ge_dict, ge_curve


# ============================================================================
# Training Functions
# ============================================================================

def pretrain_on_ches(model, train_loader, val_loader, num_epochs, learning_rate):
    """Pre-train model on CHES dataset with supervision."""
    print("\n" + "="*60)
    print("Pre-training on CHES dataset...")
    print("="*60)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for traces, labels, _ in pbar:
            traces, labels = traces.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _, _ = model(traces)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1), 'acc': 100. * correct / total})
        
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for traces, labels, _ in val_loader:
                traces, labels = traces.to(device), labels.to(device)
                outputs, _, _ = model(traces)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, f'{OUTPUT_DIR}/ches_pretrained.pth')
            print(f"  -> Saved best model")
    
    return model


def utla_train_epoch(epoch, target_model, source_model, discriminator,
                     source_loader, target_loader,
                     optimizer_encoder, optimizer_disc):
    """
    One epoch of UTLA training using ADDA+MMD (Eq. 12-15 in paper).
    """
    target_model.train()
    discriminator.train()
    source_model.eval()  # Source model is frozen
    
    criterion = nn.CrossEntropyLoss()
    
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    n_batches = min(len(source_loader), len(target_loader))
    
    total_disc_loss = 0
    total_enc_loss = 0
    total_mmd_loss = 0
    disc_acc = 0
    
    for batch_idx in range(n_batches):
        try:
            source_data, _, _ = next(source_iter)
            target_data, _, _ = next(target_iter)
        except StopIteration:
            break
        
        source_data = source_data.to(device)
        target_data = target_data.to(device)
        
        # ============================
        # Train Discriminator (Eq. 12)
        # ============================
        if USE_ADVERSARIAL:
            optimizer_disc.zero_grad()
            
            with torch.no_grad():
                _, feat_s, _ = source_model(source_data)
            _, feat_t, _ = target_model(target_data)
            
            feat_concat = torch.cat((feat_s, feat_t), 0)
            pred_concat = discriminator(feat_concat.detach())
            
            # Domain labels: 1 for source, 0 for target
            label_s = torch.ones(feat_s.size(0), dtype=torch.long, device=device)
            label_t = torch.zeros(feat_t.size(0), dtype=torch.long, device=device)
            label_concat = torch.cat((label_s, label_t), 0)
            
            loss_disc = criterion(pred_concat, label_concat)
            loss_disc.backward()
            optimizer_disc.step()
            
            # Discriminator accuracy
            pred_labels = pred_concat.argmax(dim=1)
            disc_acc += (pred_labels == label_concat).float().mean().item()
            total_disc_loss += loss_disc.item()
        
        # ============================
        # Train Encoder (Target)
        # ============================
        optimizer_encoder.zero_grad()
        
        with torch.no_grad():
            _, feat_s, feat_s_m1 = source_model(source_data)
        _, feat_t, feat_t_m1 = target_model(target_data)
        
        # MMD loss at encoder output (λ1) and penultimate layer (λ2)
        # Per Eq. 15: L*_advE = L_advE + Σ_i λ_i · MMD²(...)
        mmd_loss1 = mmd_rbf(feat_s, feat_t)      # Encoder output (both 192-dim)
        
        # Only compute penultimate MMD if dimensions match
        # (3-layer vs 4-layer models have different intermediate sizes)
        if feat_s_m1.shape[1] == feat_t_m1.shape[1]:
            mmd_loss2 = mmd_rbf(feat_s_m1, feat_t_m1)
            loss_mmd = LAMBDA1 * mmd_loss1 + LAMBDA2 * mmd_loss2
        else:
            # Skip penultimate MMD due to dimension mismatch
            mmd_loss2 = torch.tensor(0.0, device=device)
            loss_mmd = LAMBDA1 * mmd_loss1
        
        if USE_ADVERSARIAL:
            # Full ADDA+MMD: Eq. 15 in paper
            pred_t = discriminator(feat_t)
            fake_label = torch.ones(feat_t.size(0), dtype=torch.long, device=device)
            loss_adv = criterion(pred_t, fake_label)
            total_loss = loss_adv + loss_mmd  # No scaling per paper
        else:
            # MMD-only mode
            total_loss = loss_mmd
            loss_adv = torch.tensor(0.0)
        
        total_loss.backward()
        optimizer_encoder.step()
        
        total_enc_loss += loss_adv.item()
        total_mmd_loss += loss_mmd.item()
    
    avg_disc_loss = total_disc_loss / n_batches if USE_ADVERSARIAL else 0
    avg_enc_loss = total_enc_loss / n_batches if USE_ADVERSARIAL else 0
    avg_mmd_loss = total_mmd_loss / n_batches
    avg_disc_acc = disc_acc / n_batches * 100 if USE_ADVERSARIAL else 0
    
    mode_str = "" if USE_ADVERSARIAL else "[MMD-only]"
    
    print(f"Epoch {epoch} {mode_str}: Disc Loss={avg_disc_loss:.4f}, Disc Acc={avg_disc_acc:.1f}%, "
          f"Enc Loss={avg_enc_loss:.4f}, MMD Loss={avg_mmd_loss:.4f}")
    
    return avg_disc_loss, avg_enc_loss, avg_mmd_loss


def evaluate_attack(model, loader, plaintexts, real_key, model_flag='model', use_statistical=False):
    """
    Evaluate attack performance.
    
    Args:
        model: trained model
        loader: DataLoader for attack traces
        plaintexts: plaintext bytes
        real_key: correct key byte
        model_flag: name for saving results
        use_statistical: if True, use statistical GEEA (for strong signals like XMEGA)
                        if False, use empirical Monte Carlo (for weak signals like CHES)
    """
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for traces, _, _ in tqdm(loader, desc="Attack"):
            traces = traces.to(device)
            outputs, _, _ = model(traces)
            probs = torch.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    
    if use_statistical:
        # Statistical GEEA - for XMEGA (strong signals)
        ntge, ge_dict, ge_curve = compute_statistical_ge(all_preds, plaintexts, real_key)
        
        print(f"\nAttack Results ({model_flag}) [Statistical GEEA]:")
        print(f"  NTGE: {ntge}")
        print(f"  Final GE: {ge_curve[-1]:.1f}")
        
        # Return in same format as empirical for compatibility
        success_rate = 100.0 if ntge < len(ge_curve) else 0.0
        ntge_mean = ntge
    else:
        # Empirical Monte Carlo - for CHES (weak signals)
        ntge_mean, ntge_median, success_rate, ge_curve = compute_empirical_ntge(
            all_preds, plaintexts, real_key, N_EXPERIMENTS
        )
        ntge = ntge_mean
        
        print(f"\nAttack Results ({model_flag}) [Empirical MC]:")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Mean NTGE: {ntge_mean}")
        print(f"  Median NTGE: {ntge_median}")
        print(f"  Final GE: {ge_curve[-1]:.1f}")
    
    # Plot GE curve
    plt.figure(figsize=(10, 5))
    plt.semilogx(range(1, len(ge_curve)+1), ge_curve, 'b-', linewidth=1)
    plt.axhline(y=1, color='green', linestyle='--', label='GE=1')
    plt.axhline(y=128, color='gray', linestyle=':', label='GE=128')
    if ntge != float('inf') and ntge < len(ge_curve):
        plt.axvline(x=ntge, color='red', linestyle='--', label=f'NTGE={ntge}')
    plt.xlabel('Number of traces')
    plt.ylabel('Guessing Entropy')
    method = 'Statistical' if use_statistical else 'Empirical'
    plt.title(f'GE Curve - {model_flag} ({method})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{FIGURES_DIR}/GE_{model_flag}.png', dpi=150)
    plt.close()
    
    if use_statistical:
        return ntge_mean, ntge_mean, success_rate, ge_curve  # ntge_median = ntge_mean for statistical
    else:
        return ntge_mean, ntge_median, success_rate, ge_curve


# ============================================================================
# Main Functions
# ============================================================================

def run_pretrain():
    """Pre-train model on CHES dataset."""
    # Load CHES data
    if USE_PREPROCESSED:
        ches_data = load_ches_preprocessed(CHES_PREPROCESSED_PATH)
        # Preprocessed data is already windowed and standardized
        X_train = ches_data['X_train'][:CHES_TRAIN_NUM + CHES_VALID_NUM]
        X_attack = ches_data['X_attack']
        # Use offset=0 since already windowed
        ches_offset = 0
        ches_length = X_train.shape[1]  # Already the correct length
    else:
        ches_data = load_ches_data(CHES_DATA_PATH)
        # Preprocess (filter + standardization)
        X_train = preprocess_per_trace(
            ches_data['X_train'][:CHES_TRAIN_NUM + CHES_VALID_NUM],
            use_filter=USE_FILTERING, cutoff=FILTER_CUTOFF, order=FILTER_ORDER
        )
        X_attack = preprocess_per_trace(
            ches_data['X_attack'],
            use_filter=USE_FILTERING, cutoff=FILTER_CUTOFF, order=FILTER_ORDER
        )
        ches_offset = CHES_TRACE_OFFSET
        ches_length = CHES_TRACE_LENGTH
    
    # Create datasets
    train_dataset = CHESDataset(
        X_train[:CHES_TRAIN_NUM], 
        ches_data['Y_train'][:CHES_TRAIN_NUM],
        ches_data['P_train'][:CHES_TRAIN_NUM],
        ches_offset, ches_length
    )
    val_dataset = CHESDataset(
        X_train[CHES_TRAIN_NUM:CHES_TRAIN_NUM + CHES_VALID_NUM],
        ches_data['Y_train'][CHES_TRAIN_NUM:CHES_TRAIN_NUM + CHES_VALID_NUM],
        ches_data['P_train'][CHES_TRAIN_NUM:CHES_TRAIN_NUM + CHES_VALID_NUM],
        ches_offset, ches_length
    )
    
    # For attack dataset, we need Y_attack which may not be in preprocessed file
    if ches_data.get('Y_attack') is not None:
        attack_dataset = CHESDataset(
            X_attack, ches_data['Y_attack'], ches_data['P_attack'],
            ches_offset, ches_length
        )
        attack_loader = DataLoader(attack_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    else:
        attack_loader = None
        print("Note: Y_attack not available in preprocessed file, skipping attack evaluation")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model - use 4-layer if configured (for XMEGA compatibility)
    if USE_4LAYER_ARCH:
        print("Using 4-layer architecture (matches XMEGA)")
        model = UTLA_Net_4Layer(trace_length=ches_length, num_classes=CLASS_NUM)
    else:
        print("Using 3-layer architecture (original CHES)")
        model = UTLA_Net_3Layer(trace_length=ches_length, num_classes=CLASS_NUM)
    model = model.to(device)
    
    # Pre-train
    model = pretrain_on_ches(model, train_loader, val_loader, PRETRAIN_EPOCHS, PRETRAIN_LR)
    
    # Save with architecture info
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': '4layer' if USE_4LAYER_ARCH else '3layer',
        'trace_length': ches_length,
    }, CHES_PRETRAINED_PATH)
    print(f"Saved pretrained model to {CHES_PRETRAINED_PATH}")
    
    # Evaluate on CHES attack set
    if attack_loader is not None:
        print("\nEvaluating on CHES attack set...")
        evaluate_attack(model, attack_loader, ches_data['P_attack'], 
                       ches_data['correct_key'], 'ches_pretrained')
    else:
        print("\nSkipping CHES attack evaluation (no Y_attack in preprocessed file)")


def run_transfer(xmega_device=XMEGA_TARGET_DEVICE):
    """Transfer pre-trained CHES model to XMEGA."""
    
    # Check for existing checkpoints with mismatched architecture
    current_arch = '4layer' if USE_4LAYER_ARCH else '3layer'
    model_path = f'{OUTPUT_DIR}/utla_ches_to_xmega{xmega_device}.pth'
    baseline_path = f'{OUTPUT_DIR}/utla_ches_to_xmega{xmega_device}_baseline.pth'
    
    for ckpt_path in [model_path, baseline_path]:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            ckpt_arch = ckpt.get('architecture', '3layer')
            if ckpt_arch != current_arch:
                print(f"\n{'='*60}")
                print(f"WARNING: Existing checkpoint has mismatched architecture!")
                print(f"  Checkpoint: {ckpt_path}")
                print(f"  Checkpoint arch: {ckpt_arch}, Current config: {current_arch}")
                print(f"  Deleting old checkpoint to avoid conflicts...")
                print(f"{'='*60}\n")
                os.remove(ckpt_path)
    
    # Load CHES data (source)
    print("Loading source domain (CHES)...")
    if USE_PREPROCESSED:
        ches_data = load_ches_preprocessed(CHES_PREPROCESSED_PATH)
        X_ches = ches_data['X_train'][:CHES_TRAIN_NUM]
        ches_offset = 0
        ches_length = X_ches.shape[1]
    else:
        ches_data = load_ches_data(CHES_DATA_PATH)
        X_ches = preprocess_per_trace(
            ches_data['X_train'][:CHES_TRAIN_NUM],
            use_filter=USE_FILTERING, cutoff=FILTER_CUTOFF, order=FILTER_ORDER
        )
        ches_offset = CHES_TRACE_OFFSET
        ches_length = CHES_TRACE_LENGTH
    
    ches_dataset = CHESDataset(
        X_ches, ches_data['Y_train'][:CHES_TRAIN_NUM],
        ches_data['P_train'][:CHES_TRAIN_NUM],
        ches_offset, ches_length
    )
    ches_loader = DataLoader(ches_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, drop_last=True)
    
    # Load XMEGA data (target)
    # Note: XMEGA traces are already cleaner (power traces), so we don't apply filtering
    print(f"\nLoading target domain (XMEGA device {xmega_device})...")
    xmega_data = load_xmega_data(XMEGA_DATA_PATH, xmega_device)
    X_xmega_train = preprocess_per_trace(xmega_data['X_train'][:XMEGA_TRAIN_NUM], use_filter=False)
    X_xmega_attack = preprocess_per_trace(xmega_data['X_attack'][:XMEGA_TEST_NUM], use_filter=False)
    
    xmega_train_dataset = XMEGADataset(
        X_xmega_train, xmega_data['Y_train'][:XMEGA_TRAIN_NUM],
        xmega_data['P_train'][:XMEGA_TRAIN_NUM],
        XMEGA_TRACE_OFFSET, XMEGA_TRACE_LENGTH
    )
    xmega_attack_dataset = XMEGADataset(
        X_xmega_attack, xmega_data['Y_attack'][:XMEGA_TEST_NUM],
        xmega_data['P_attack'][:XMEGA_TEST_NUM],
        XMEGA_TRACE_OFFSET, XMEGA_TRACE_LENGTH
    )
    
    xmega_train_loader = DataLoader(xmega_train_dataset, batch_size=BATCH_SIZE, 
                                    shuffle=True, num_workers=4, drop_last=True)
    xmega_attack_loader = DataLoader(xmega_attack_dataset, batch_size=BATCH_SIZE, 
                                     shuffle=False, num_workers=4)
    
    # Create models - use 4-layer architecture for both (matches XMEGA)
    if USE_4LAYER_ARCH:
        print("Using 4-layer architecture for both source and target models")
        source_model = UTLA_Net_4Layer(trace_length=ches_length, num_classes=CLASS_NUM).to(device)
        target_model = UTLA_Net_4Layer(trace_length=XMEGA_TRACE_LENGTH, num_classes=CLASS_NUM).to(device)
    else:
        print("Using 3-layer architecture")
        source_model = UTLA_Net_3Layer(trace_length=ches_length, num_classes=CLASS_NUM).to(device)
        target_model = UTLA_Net_3Layer(trace_length=XMEGA_TRACE_LENGTH, num_classes=CLASS_NUM).to(device)
    discriminator = Discriminator(input_size=source_model.flatten_size).to(device)
    
    # Initialize models based on ENCODER_INIT setting
    if ENCODER_INIT == 'random':
        # Load source model, random target encoder + source classifier
        print("Using RANDOM initialization for target encoder")
        if os.path.exists(CHES_PRETRAINED_PATH):
            checkpoint = torch.load(CHES_PRETRAINED_PATH)
            source_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded source model from {CHES_PRETRAINED_PATH}")
        else:
            print(f"  WARNING: {CHES_PRETRAINED_PATH} not found, using random source model")
            print(f"  Run with --mode pretrain first to create it")
        target_model.classifier_1.load_state_dict(source_model.classifier_1.state_dict())
        target_model.final_classifier.load_state_dict(source_model.final_classifier.state_dict())
        
    elif ENCODER_INIT == 'source':
        # Full source initialization (homogeneous transfer)
        print("Using SOURCE (CHES) initialization for target encoder")
        if os.path.exists(CHES_PRETRAINED_PATH):
            checkpoint = torch.load(CHES_PRETRAINED_PATH)
            source_model.load_state_dict(checkpoint['model_state_dict'])
            target_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded both models from {CHES_PRETRAINED_PATH}")
        else:
            print(f"  WARNING: {CHES_PRETRAINED_PATH} not found")
            print(f"  Run with --mode pretrain first to create it")
            
    elif ENCODER_INIT == 'xmega':
        # Use supervised XMEGA model for BOTH source and target (diagnostic)
        print(f"Using XMEGA SUPERVISED initialization")
        print(f"  Loading from: {XMEGA_SUPERVISED_PATH}")
        xmega_checkpoint = torch.load(XMEGA_SUPERVISED_PATH)
        xmega_state = xmega_checkpoint['model_state_dict']
        
        # Load XMEGA model into both source and target
        # This tests: does ADDA+MMD preserve a good encoder?
        source_model.load_state_dict(xmega_state)
        target_model.load_state_dict(xmega_state)
        
        print("  Loaded XMEGA supervised model into BOTH source and target")
        print("  This tests if ADDA+MMD preserves a good encoder")
    else:
        raise ValueError(f"Unknown ENCODER_INIT: {ENCODER_INIT}")
    
    # Freeze source model
    for param in source_model.parameters():
        param.requires_grad = False
    
    # Freeze classifier in target model (only train encoder)
    for param in target_model.classifier_1.parameters():
        param.requires_grad = False
    for param in target_model.final_classifier.parameters():
        param.requires_grad = False
    
    # Optimizers - encoder layers only (classifier is frozen)
    # Dynamically include features_4 if it exists (4-layer model)
    encoder_params = [
        {'params': target_model.features_1.parameters()},
        {'params': target_model.features_2.parameters()},
        {'params': target_model.features_3.parameters()},
    ]
    if hasattr(target_model, 'features_4'):
        encoder_params.append({'params': target_model.features_4.parameters()})
        print("  Optimizer includes features_4 (4-layer model)")
    
    optimizer_encoder = optim.SGD(encoder_params, lr=ENCODER_LR, weight_decay=0.0005, momentum=0.9)
    
    optimizer_disc = optim.SGD(
        discriminator.parameters(),
        lr=DISCRIMINATOR_LR, weight_decay=0.0005, momentum=0.9
    )
    
    # Evaluate before transfer (get baseline)
    print("\n" + "="*60)
    print("Evaluation BEFORE transfer:")
    print("="*60)
    baseline_ntge, _, _, _ = evaluate_attack(target_model, xmega_attack_loader, 
                   xmega_data['P_attack'][:XMEGA_TEST_NUM],
                   xmega_data['correct_key'], 'before_transfer', use_statistical=True)
    
    # Save baseline model
    torch.save({
        'epoch': 0,
        'model_state_dict': target_model.state_dict(),
        'ntge': baseline_ntge,
        'architecture': '4layer' if USE_4LAYER_ARCH else '3layer',
    }, f'{OUTPUT_DIR}/utla_ches_to_xmega{xmega_device}_baseline.pth')
    print(f"Baseline NTGE: {baseline_ntge}")
    
    # UTLA Training
    print("\n" + "="*60)
    print("UTLA Transfer Training...")
    print(f"  Mode: {'ADDA + MMD' if USE_ADVERSARIAL else 'MMD-only'}")
    print(f"  Architecture: {'4-layer' if USE_4LAYER_ARCH else '3-layer'}")
    print(f"  Encoder init: {ENCODER_INIT.upper()}")
    print(f"  Total epochs: {TRANSFER_EPOCHS}")
    print(f"  Encoder LR: {ENCODER_LR}, Discriminator LR: {DISCRIMINATOR_LR}")
    print(f"  Lambda MMD1 (output): {LAMBDA1}, Lambda MMD2 (penultimate): {LAMBDA2}")
    
    # Check feature dimensions
    with torch.no_grad():
        dummy_s = torch.zeros(1, 1, ches_length).to(device)
        dummy_t = torch.zeros(1, 1, XMEGA_TRACE_LENGTH).to(device)
        _, feat_s, feat_s_m1 = source_model(dummy_s)
        _, feat_t, feat_t_m1 = target_model(dummy_t)
        print(f"  Feature dims - Output: source={feat_s.shape[1]}, target={feat_t.shape[1]}")
        print(f"  Feature dims - Penultimate: source={feat_s_m1.shape[1]}, target={feat_t_m1.shape[1]}")
        if feat_s_m1.shape[1] != feat_t_m1.shape[1]:
            print(f"  WARNING: Penultimate layer dimension mismatch!")
            print(f"    Skipping λ2 MMD loss, using only λ1 (output layer)")
        else:
            print(f"  ✓ All dimensions match - full MMD loss will be used")
    print("="*60)
    
    history = {'disc_loss': [], 'enc_loss': [], 'mmd_loss': [], 'ntge': [(0, baseline_ntge)]}
    
    # Phase 1: First 10 epochs
    print("\n" + "-"*60)
    print("PHASE 1: Epochs 1-10")
    print("-"*60)
    
    for epoch in range(1, 11):
        disc_loss, enc_loss, mmd_loss = utla_train_epoch(
            epoch, target_model, source_model, discriminator,
            ches_loader, xmega_train_loader,
            optimizer_encoder, optimizer_disc
        )
        
        history['disc_loss'].append(disc_loss)
        history['enc_loss'].append(enc_loss)
        history['mmd_loss'].append(mmd_loss)
        
        # Evaluate every 2 epochs
        if epoch % 2 == 0:
            ntge, _, _, _ = evaluate_attack(
                target_model, xmega_attack_loader,
                xmega_data['P_attack'][:XMEGA_TEST_NUM],
                xmega_data['correct_key'], f'epoch{epoch}', use_statistical=True
            )
            history['ntge'].append((epoch, ntge))
    
    # Save encoder after Phase 1
    phase1_ntge = history['ntge'][-1][1]
    torch.save({
        'epoch': 10,
        'model_state_dict': target_model.state_dict(),
        'ntge': phase1_ntge,
        'architecture': '4layer' if USE_4LAYER_ARCH else '3layer',
        'phase': 1,
    }, f'{OUTPUT_DIR}/utla_ches_to_xmega{xmega_device}_phase1.pth')
    print(f"\n{'='*60}")
    print(f"PHASE 1 COMPLETE: Saved encoder at epoch 10 (NTGE={phase1_ntge})")
    print(f"{'='*60}")
    
    # Phase 2: Next 10 epochs
    print("\n" + "-"*60)
    print("PHASE 2: Epochs 11-20")
    print("-"*60)
    
    for epoch in range(11, 21):
        disc_loss, enc_loss, mmd_loss = utla_train_epoch(
            epoch, target_model, source_model, discriminator,
            ches_loader, xmega_train_loader,
            optimizer_encoder, optimizer_disc
        )
        
        history['disc_loss'].append(disc_loss)
        history['enc_loss'].append(enc_loss)
        history['mmd_loss'].append(mmd_loss)
        
        # Evaluate every 2 epochs
        if epoch % 2 == 0:
            ntge, _, _, _ = evaluate_attack(
                target_model, xmega_attack_loader,
                xmega_data['P_attack'][:XMEGA_TEST_NUM],
                xmega_data['correct_key'], f'epoch{epoch}', use_statistical=True
            )
            history['ntge'].append((epoch, ntge))
    
    # Save final encoder after Phase 2
    phase2_ntge = history['ntge'][-1][1]
    torch.save({
        'epoch': 20,
        'model_state_dict': target_model.state_dict(),
        'ntge': phase2_ntge,
        'architecture': '4layer' if USE_4LAYER_ARCH else '3layer',
        'phase': 2,
    }, f'{OUTPUT_DIR}/utla_ches_to_xmega{xmega_device}_phase2.pth')
    print(f"\n{'='*60}")
    print(f"PHASE 2 COMPLETE: Saved encoder at epoch 20 (NTGE={phase2_ntge})")
    print(f"{'='*60}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['disc_loss'], label='Discriminator')
    axes[0].plot(history['enc_loss'], label='Encoder (Adv)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Losses')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['mmd_loss'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MMD Loss')
    axes[1].set_title('MMD Loss')
    axes[1].grid(True, alpha=0.3)
    
    if history['ntge']:
        epochs, ntges = zip(*history['ntge'])
        axes[2].plot(epochs, ntges, 'bo-')
        axes[2].axhline(y=baseline_ntge, color='green', linestyle='--', label=f'Baseline={baseline_ntge}')
        axes[2].axvline(x=10, color='red', linestyle=':', label='Phase 1 end')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('NTGE')
        axes[2].set_title('NTGE over Training')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/utla_training_xmega{xmega_device}.png', dpi=150)
    plt.close()
    
    # Final Summary - report FINAL values, not best
    print("\n" + "="*60)
    print("FINAL RESULTS (reporting final NTGE, not best)")
    print("="*60)
    
    print(f"\nNTGE Progression:")
    print(f"  Baseline (epoch 0):  {baseline_ntge}")
    print(f"  Phase 1 (epoch 10):  {phase1_ntge}")
    print(f"  Phase 2 (epoch 20):  {phase2_ntge}")
    
    print(f"\nDegradation:")
    print(f"  After Phase 1: {phase1_ntge - baseline_ntge:+d} traces")
    print(f"  After Phase 2: {phase2_ntge - baseline_ntge:+d} traces")
    
    print(f"\nSaved checkpoints:")
    print(f"  Phase 1: {OUTPUT_DIR}/utla_ches_to_xmega{xmega_device}_phase1.pth")
    print(f"  Phase 2: {OUTPUT_DIR}/utla_ches_to_xmega{xmega_device}_phase2.pth")


def run_eval(xmega_device=XMEGA_TARGET_DEVICE):
    """Evaluate transferred model on XMEGA."""
    # Load XMEGA data
    xmega_data = load_xmega_data(XMEGA_DATA_PATH, xmega_device)
    X_xmega_attack = preprocess_per_trace(xmega_data['X_attack'], use_filter=False)
    
    xmega_attack_dataset = XMEGADataset(
        X_xmega_attack, xmega_data['Y_attack'],
        xmega_data['P_attack'],
        XMEGA_TRACE_OFFSET, XMEGA_TRACE_LENGTH
    )
    xmega_attack_loader = DataLoader(xmega_attack_dataset, batch_size=BATCH_SIZE, 
                                     shuffle=False, num_workers=4)
    
    # Load model
    model = UTLA_Net(trace_length=XMEGA_TRACE_LENGTH, num_classes=CLASS_NUM).to(device)
    checkpoint = torch.load(f'{OUTPUT_DIR}/utla_ches_to_xmega{xmega_device}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    evaluate_attack(model, xmega_attack_loader,
                   xmega_data['P_attack'],
                   xmega_data['correct_key'], f'eval_xmega{xmega_device}', use_statistical=True)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='UTLA: CHES to XMEGA Transfer')
    parser.add_argument('--mode', type=str, choices=['pretrain', 'transfer', 'eval'],
                       default='transfer', help='Mode: pretrain, transfer, or eval')
    parser.add_argument('--device', type=int, default=XMEGA_TARGET_DEVICE,
                       help='Target XMEGA device (1-8)')
    args = parser.parse_args()
    
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'pretrain':
        run_pretrain()
    elif args.mode == 'transfer':
        run_transfer(args.device)
    elif args.mode == 'eval':
        run_eval(args.device)


if __name__ == "__main__":
    main()
