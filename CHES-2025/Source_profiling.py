#!/usr/bin/env python
# coding: utf-8
"""
Training script for CHES Challenge 2025 using UTLA 4-Layer CNN architecture.
Uses empirical Monte Carlo NTGE computation (accurate for weak signals).

Architecture matches XMEGA pre-trained model exactly for transfer learning compatibility.

Usage:
    python Source_profiling.py
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (must be before pyplot import)

import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

# Paths - UPDATE THESE
CHES_DATA_PATH = "./CHES_Challenge.h5"
OUTPUT_DIR = "./models"
FIGURES_DIR = "./figures"
RESULTS_DIR = "./results"

# Dataset parameters
TRAIN_NUM = 400000
VALID_NUM = 50000
TEST_NUM = 100000

# Trace parameters - Window must give flatten_size=192 for XMEGA architecture compatibility
# CPA peak found at sample ~1458, so center window around it
TRACE_OFFSET = 1200   # Window [1200:1700] captures peak at ~1458
TRACE_LENGTH = 500    # Window size (gives flatten_size=192 with 4-layer arch)

# Training parameters
BATCH_SIZE = 256
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
LOG_INTERVAL = 100

# Preprocessing parameters
USE_DENOISING = True          # Apply PCA denoising
PCA_VARIANCE_RATIO = 0.95     # Keep 95% of variance for denoising
USE_STANDARDIZATION = True    # Apply global standardization

# Attack parameters
TRACE_NUM_MAX = 100000  # For GE computation - use all attack traces to find true NTGE
CLASS_NUM = 256
N_EXPERIMENTS = 100  # Number of Monte Carlo experiments for NTGE

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
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
# Preprocessing Functions
# ============================================================================

def compute_denoising_matrix(X_profiling, n_components=0.95):
    """
    Compute denoising matrix using PCA on profiling traces.
    
    Args:
        X_profiling: Profiling traces (n_traces, n_samples)
        n_components: Number of PCA components or variance ratio (0-1) to keep
    
    Returns:
        denoising_matrix: Matrix to project traces for denoising
    """
    print("Computing denoising matrix...")
    
    # Use a subset for PCA if dataset is very large
    max_traces_for_pca = 50000
    if len(X_profiling) > max_traces_for_pca:
        indices = np.random.choice(len(X_profiling), max_traces_for_pca, replace=False)
        X_subset = X_profiling[indices].astype(np.float32)
    else:
        X_subset = X_profiling.astype(np.float32)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_subset)
    
    # Denoising matrix: project to PCA space and back
    # denoised = X @ V.T @ V where V are the principal components
    components = pca.components_  # (n_components, n_features)
    denoising_matrix = components.T @ components  # (n_features, n_features)
    
    print(f"  PCA components: {pca.n_components_}")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    print(f"  Denoising matrix shape: {denoising_matrix.shape}")
    
    return denoising_matrix.astype(np.float32)


def denoise_traces(X, denoising_matrix):
    """Apply denoising matrix to traces."""
    print("Denoising traces...")
    X_denoised = X @ denoising_matrix
    return X_denoised


def compute_standardization_params(X_profiling):
    """
    Compute mean and std from profiling traces for standardization.
    
    Args:
        X_profiling: Profiling traces (n_traces, n_samples)
    
    Returns:
        mean: Per-sample mean (n_samples,)
        std: Per-sample std (n_samples,)
    """
    print("Computing standardization parameters...")
    mean = np.mean(X_profiling, axis=0)
    std = np.std(X_profiling, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return mean, std


def standardize_traces(X, mean, std):
    """Apply standardization using precomputed mean/std."""
    print("Standardizing traces...")
    return (X - mean) / std


def preprocess_traces(X, denoising_matrix=None, mean=None, std=None):
    """
    Full preprocessing pipeline.
    
    Args:
        X: Raw traces
        denoising_matrix: PCA denoising matrix (optional)
        mean: Standardization mean (optional)
        std: Standardization std (optional)
    
    Returns:
        X_processed: Preprocessed traces
    """
    X_processed = X.astype(np.float32)
    
    # Step 1: Denoising (if matrix provided)
    if denoising_matrix is not None:
        X_processed = denoise_traces(X_processed, denoising_matrix)
    
    # Step 2: Standardization (if params provided)
    if mean is not None and std is not None:
        X_processed = standardize_traces(X_processed, mean, std)
    
    return X_processed


# ============================================================================
# Dataset Class
# ============================================================================

class CHESDataset(Dataset):
    """Dataset class for CHES Challenge 2025."""
    
    def __init__(self, traces, labels, plaintexts, keys, trace_offset, trace_length):
        self.traces = traces
        self.labels = labels
        self.plaintexts = plaintexts
        self.keys = keys
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
# CNN Model (UTLA 4-Layer Architecture - matches XMEGA exactly)
# ============================================================================

class UTLA_Net(nn.Module):
    """
    UTLA 4-layer CNN architecture matching XMEGA power analysis model.
    
    This architecture is identical to the pre-trained XMEGA model, enabling
    direct weight transfer for domain adaptation experiments.
    
    Architecture:
        features_1: Conv(1→8, k=1), Pool(2,2)   → (8, 250)
        features_2: Conv(8→16, k=9), Pool(9,9)  → (16, 26)
        features_3: Conv(16→32, k=2), Pool(3,3) → (32, 8)
        features_4: Conv(32→64, k=2), Pool(2,2) → Flatten → 192
        classifier_1: Linear(192→2)
        final_classifier: Linear(2→256)
    """
    
    def __init__(self, trace_length, num_classes=256):
        super(UTLA_Net, self).__init__()
        
        self.trace_length = trace_length
        
        # Encoder - 4 layers (matches XMEGA exactly)
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
        
        # Classifier
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
        
        # Verify compatibility with XMEGA classifier (must be 192)
        if self.flatten_size != 192:
            print(f"WARNING: flatten_size={self.flatten_size} != 192.")
            print(f"  Adjust trace_length for XMEGA compatibility.")
            print(f"  Expected 192 for transfer learning with XMEGA model.")
    
    def forward(self, x):
        x = self.features_1(x)
        x = self.features_2(x)
        x = self.features_3(x)
        x = self.features_4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_1(x)
        output = self.final_classifier(x)
        return output
    
    def forward_features(self, x):
        """
        Forward pass returning intermediate features for transfer learning.
        
        Returns:
            output: Classification logits (batch, 256)
            feat: Encoder output features (batch, 192)
            feat_m1: Penultimate layer features (batch, 256) - from features_3
        """
        x = self.features_1(x)
        x = self.features_2(x)
        x = self.features_3(x)
        feat_m1 = x.view(x.size(0), -1)  # Penultimate: (batch, 32*8=256)
        
        x = self.features_4(x)
        feat = x.view(x.size(0), -1)  # Encoder output: (batch, 192)
        
        x = self.classifier_1(feat)
        output = self.final_classifier(x)
        
        return output, feat, feat_m1


# ============================================================================
# Guessing Entropy (Empirical Monte Carlo - accurate for weak signals)
# ============================================================================

def compute_empirical_ntge(preds, plaintexts, real_key, n_experiments=100, model_flag='model'):
    """
    Compute NTGE using empirical Monte Carlo method.
    
    This method:
    1. Precomputes log-probabilities for all traces and key guesses
    2. Runs multiple experiments with random trace orderings
    3. Tracks when correct key first reaches rank 1
    4. Averages NTGE across successful experiments
    
    This is accurate for weak signals where analytical GEEA fails.
    
    Args:
        preds: numpy array (N, 256) - model output probabilities
        plaintexts: numpy array (N,) - plaintext bytes
        real_key: int - correct key byte
        n_experiments: int - number of Monte Carlo experiments
        model_flag: str - name for saving results
    
    Returns:
        ntge: mean NTGE across experiments
        ntge_median: median NTGE
        success_rate: percentage of experiments where attack succeeded
        ge_curve: mean GE curve across experiments
    """
    n_traces = min(TRACE_NUM_MAX, len(preds))
    preds = preds[:n_traces].astype(np.float64)
    plaintexts = plaintexts[:n_traces]
    
    # Clip probabilities and take log
    eps = 1e-40
    preds = np.clip(preds, eps, 1.0)
    log_preds = np.log(preds)
    
    # Precompute label lookup table: label[plaintext, key] = Sbox[pt ^ k]
    pt_range = np.arange(256, dtype=np.uint8)
    key_range = np.arange(256, dtype=np.uint8)
    label_table = Sbox[pt_range[:, None] ^ key_range[None, :]]  # (256, 256)
    
    # For each trace, get the label for each key guess
    guess_labels = label_table[plaintexts, :]  # (n_traces, 256)
    
    # Get log-probability for each trace and key hypothesis
    trace_idx = np.arange(n_traces)[:, None]
    log_probs_per_key = log_preds[trace_idx, guess_labels]  # (n_traces, 256)
    
    # Run Monte Carlo experiments
    ntge_list = []
    ge_curves = []
    
    print(f"Running {n_experiments} Monte Carlo experiments...")
    for exp in tqdm(range(n_experiments), desc="MC Experiments"):
        # Random permutation of traces
        perm = np.random.permutation(n_traces)
        log_probs_shuffled = log_probs_per_key[perm, :]  # (n_traces, 256)
        
        # Cumulative sum of log-probabilities for each key
        cumsum = np.cumsum(log_probs_shuffled, axis=0)  # (n_traces, 256)
        
        # Rank of correct key at each trace count
        # Rank = 1 + number of keys with higher cumsum
        correct_cumsum = cumsum[:, real_key]  # (n_traces,)
        keys_beating_correct = np.sum(cumsum > correct_cumsum[:, None], axis=1)  # (n_traces,)
        ge_curve = 1 + keys_beating_correct  # Rank of correct key
        ge_curves.append(ge_curve)
        
        # Find NTGE (first trace where rank = 1)
        indices = np.where(ge_curve == 1)[0]
        if len(indices) > 0:
            ntge_list.append(indices[0] + 1)
    
    # Compute statistics
    mean_ge_curve = np.mean(ge_curves, axis=0)
    
    if len(ntge_list) > 0:
        ntge_mean = int(np.mean(ntge_list))
        ntge_median = int(np.median(ntge_list))
        ntge_std = np.std(ntge_list)
        success_rate = len(ntge_list) / n_experiments * 100
    else:
        ntge_mean = float('inf')
        ntge_median = float('inf')
        ntge_std = 0
        success_rate = 0
    
    # Print results
    print(f"\nEmpirical NTGE Results ({n_experiments} experiments):")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Mean NTGE: {ntge_mean}")
    print(f"  Median NTGE: {ntge_median}")
    if len(ntge_list) > 0:
        print(f"  Std NTGE: {ntge_std:.1f}")
        print(f"  Min NTGE: {min(ntge_list)}")
        print(f"  Max NTGE: {max(ntge_list)}")
    
    # GE at sample points
    sample_points = [i for i in [1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000] 
                     if i <= n_traces]
    print(f"\nMean GE at sample points:")
    for n in sample_points:
        print(f"  GE[{n:6d}] = {mean_ge_curve[n-1]:.1f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Mean GE curve
    axes[0].semilogx(range(1, n_traces+1), mean_ge_curve, 'b-', linewidth=1, alpha=0.8)
    axes[0].axhline(y=1, color='green', linestyle='--', label='GE=1 (success)')
    axes[0].axhline(y=128, color='gray', linestyle=':', label='GE=128 (random)')
    if ntge_mean != float('inf'):
        axes[0].axvline(x=ntge_mean, color='red', linestyle='--', label=f'NTGE={ntge_mean}')
    axes[0].set_xlabel('Number of traces (log scale)', fontsize=12)
    axes[0].set_ylabel('Mean Guessing Entropy', fontsize=12)
    axes[0].set_title(f'Empirical GE Curve ({n_experiments} experiments)', fontsize=14)
    axes[0].set_ylim(0, 260)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: NTGE distribution
    if len(ntge_list) > 0:
        axes[1].hist(ntge_list, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=ntge_mean, color='red', linestyle='--', linewidth=2, label=f'Mean={ntge_mean}')
        axes[1].axvline(x=ntge_median, color='orange', linestyle='--', linewidth=2, label=f'Median={ntge_median}')
        axes[1].set_xlabel('NTGE', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title(f'NTGE Distribution (Success rate: {success_rate:.1f}%)', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No successful attacks', ha='center', va='center', fontsize=14)
        axes[1].set_title('NTGE Distribution (0% success)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/GE_empirical_{model_flag}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    results = {
        'ntge_mean': ntge_mean,
        'ntge_median': ntge_median,
        'ntge_std': ntge_std if len(ntge_list) > 0 else 0,
        'success_rate': success_rate,
        'ntge_list': ntge_list,
        'mean_ge_curve': mean_ge_curve,
        'n_experiments': n_experiments,
    }
    np.save(f'{RESULTS_DIR}/GE_empirical_{model_flag}.npy', results, allow_pickle=True)
    
    return ntge_mean, ntge_median, success_rate, mean_ge_curve


def plot_guessing_entropy(preds, plaintexts, real_key, model_flag='model'):
    """
    Compute GE using empirical Monte Carlo method.
    
    This is a wrapper that calls compute_empirical_ntge and returns
    values in the same format as the original analytical function.
    
    Args:
        preds: numpy array (N, 256) - model output probabilities
        plaintexts: numpy array (N,) - plaintext bytes
        real_key: int - correct key byte
        model_flag: str - name for saving results
    
    Returns:
        ntge: estimated NTGE (mean)
        final_ge: GE at trace_num_max
        extrapolated_ntge: same as ntge_median (no extrapolation needed for empirical)
    """
    ntge_mean, ntge_median, success_rate, mean_ge_curve = compute_empirical_ntge(
        preds, plaintexts, real_key, 
        n_experiments=N_EXPERIMENTS,
        model_flag=model_flag
    )
    
    # Final GE is the last value in the mean GE curve
    final_ge = int(mean_ge_curve[-1])
    
    # For empirical method, there's no extrapolation - NTGE is directly measured
    # Use median as it's more robust to outliers
    extrapolated_ntge = ntge_median
    
    return ntge_mean, final_ge, extrapolated_ntge


# ============================================================================
# Data Loading
# ============================================================================

def load_ches_data(filepath, byte=0):
    """Load CHES Challenge dataset."""
    print(f"Loading CHES dataset from {filepath}...")
    
    with h5py.File(filepath, "r") as f:
        # Profiling traces
        X_profiling = np.array(f['Profiling_traces/traces'])
        P_profiling = np.array(f['Profiling_traces/metadata'][:]['plaintext'][:, byte])
        K_profiling = np.array(f['Profiling_traces/metadata'][:]['key'][:, byte])
        
        if byte == 0:
            Y_profiling = np.array(f['Profiling_traces/metadata'][:]['labels'])
        else:
            Y_profiling = np.array([Sbox[p ^ k] for p, k in zip(P_profiling, K_profiling)])
        
        # Attack traces
        X_attack = np.array(f['Attack_traces/traces'])
        P_attack = np.array(f['Attack_traces/metadata'][:]['plaintext'][:, byte])
        K_attack = np.array(f['Attack_traces/metadata'][:]['key'][:, byte])
        
        if byte == 0:
            Y_attack = np.array(f['Attack_traces/metadata'][:]['labels'])
        else:
            Y_attack = np.array([Sbox[p ^ k] for p, k in zip(P_attack, K_attack)])
    
    print(f"Profiling: {X_profiling.shape}, Attack: {X_attack.shape}")
    print(f"Correct key: {K_attack[0]}")
    
    return (X_profiling, Y_profiling, P_profiling, K_profiling), \
           (X_attack, Y_attack, P_attack, K_attack)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (traces, labels, _) in enumerate(pbar):
        traces = traces.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(traces)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % LOG_INTERVAL == 0:
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for traces, labels, _ in loader:
            traces = traces.to(device)
            labels = labels.to(device)
            
            outputs = model(traces)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


def attack(model, loader, plaintexts, real_key, model_flag='model'):
    """Run attack and compute GE/NTGE."""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for traces, _, _ in tqdm(loader, desc="Attack"):
            traces = traces.to(device)
            outputs = model(traces)
            probs = torch.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    
    ntge, final_ge, extrapolated_ntge = plot_guessing_entropy(all_preds, plaintexts, real_key, model_flag)
    
    return ntge, final_ge, extrapolated_ntge


# ============================================================================
# Main
# ============================================================================

def main():
    print(f"Device: {device}")
    print("="*60)
    print("UTLA 4-Layer Architecture (XMEGA Compatible)")
    print("="*60)
    
    # Load data
    (X_prof, Y_prof, P_prof, K_prof), (X_attack, Y_attack, P_attack, K_attack) = \
        load_ches_data(CHES_DATA_PATH)
    
    correct_key = K_attack[0]
    
    # =========================================================================
    # Preprocessing Pipeline
    # =========================================================================
    print("\n" + "="*60)
    print("Preprocessing...")
    print("="*60)
    
    X_prof_processed = X_prof.astype(np.float32)
    X_attack_processed = X_attack.astype(np.float32)
    
    denoising_matrix = None
    std_mean = None
    std_std = None
    
    # Step 1: Denoising (optional)
    if USE_DENOISING:
        denoising_matrix = compute_denoising_matrix(X_prof, n_components=PCA_VARIANCE_RATIO)
        X_prof_processed = denoise_traces(X_prof_processed, denoising_matrix)
        X_attack_processed = denoise_traces(X_attack_processed, denoising_matrix)
    else:
        print("Skipping denoising (USE_DENOISING=False)")
    
    # Step 2: Standardization (optional)
    if USE_STANDARDIZATION:
        std_mean, std_std = compute_standardization_params(X_prof_processed)
        X_prof_processed = standardize_traces(X_prof_processed, std_mean, std_std)
        X_attack_processed = standardize_traces(X_attack_processed, std_mean, std_std)
    else:
        print("Skipping standardization (USE_STANDARDIZATION=False)")
    
    # Step 3: Save preprocessing params for inference
    preprocessing_params = {
        'denoising_matrix': denoising_matrix,
        'mean': std_mean,
        'std': std_std,
        'trace_offset': TRACE_OFFSET,
        'trace_length': TRACE_LENGTH,
        'use_denoising': USE_DENOISING,
        'use_standardization': USE_STANDARDIZATION,
        'architecture': '4layer',
    }
    np.save(f'{OUTPUT_DIR}/preprocessing_params.npy', preprocessing_params, allow_pickle=True)
    print(f"Saved preprocessing params to {OUTPUT_DIR}/preprocessing_params.npy")
    
    # Free memory
    del X_prof, X_attack
    
    # =========================================================================
    # Split and Create Datasets
    # =========================================================================
    print("\n" + "="*60)
    print("Creating datasets...")
    print("="*60)
    
    # Split profiling data
    X_train, Y_train = X_prof_processed[:TRAIN_NUM], Y_prof[:TRAIN_NUM]
    P_train, K_train = P_prof[:TRAIN_NUM], K_prof[:TRAIN_NUM]
    
    X_val, Y_val = X_prof_processed[TRAIN_NUM:TRAIN_NUM+VALID_NUM], Y_prof[TRAIN_NUM:TRAIN_NUM+VALID_NUM]
    P_val, K_val = P_prof[TRAIN_NUM:TRAIN_NUM+VALID_NUM], K_prof[TRAIN_NUM:TRAIN_NUM+VALID_NUM]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Attack: {len(X_attack_processed)}")
    
    # Create datasets
    train_dataset = CHESDataset(X_train, Y_train, P_train, K_train, TRACE_OFFSET, TRACE_LENGTH)
    val_dataset = CHESDataset(X_val, Y_val, P_val, K_val, TRACE_OFFSET, TRACE_LENGTH)
    attack_dataset = CHESDataset(X_attack_processed, Y_attack, P_attack, K_attack, TRACE_OFFSET, TRACE_LENGTH)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    attack_loader = DataLoader(attack_dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=4, pin_memory=True)
    
    # Create model
    print("\nCreating 4-layer UTLA model (XMEGA compatible)...")
    model = UTLA_Net(trace_length=TRACE_LENGTH, num_classes=CLASS_NUM)
    model = model.to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    print(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training
    print("\n" + "="*60)
    print("Training...")
    print("="*60)
    
    best_val_loss = float('inf')
    best_ntge = float('inf')
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'ntge': []}
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        # Attack every 5 epochs
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            ntge, final_ge, extrapolated_ntge = attack(model, attack_loader, P_attack, correct_key, f'epoch{epoch}')
            history['ntge'].append((epoch, ntge, extrapolated_ntge))
            
            # Use mean NTGE for comparison
            if ntge < best_ntge:
                best_ntge = ntge
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ntge': ntge,
                    'ntge_median': extrapolated_ntge,
                    'architecture': '4layer',
                    'trace_length': TRACE_LENGTH,
                }, f'{OUTPUT_DIR}/best_ntge_model.pth')
                print(f"  -> Saved best NTGE model (Mean NTGE={ntge}, Median={extrapolated_ntge})")
        
        # Save best val loss model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'architecture': '4layer',
            }, f'{OUTPUT_DIR}/best_val_model.pth')
    
    # Save training history
    np.save(f'{RESULTS_DIR}/training_history.npy', history, allow_pickle=True)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    if history['ntge']:
        epochs, ntges, medians = zip(*history['ntge'])
        axes[2].plot(epochs, ntges, 'bo-', label='Mean NTGE')
        axes[2].plot(epochs, medians, 'ro-', label='Median NTGE')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('NTGE')
        axes[2].set_title('NTGE over Training')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    checkpoint = torch.load(f'{OUTPUT_DIR}/best_ntge_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    ntge, final_ge, extrapolated_ntge = attack(model, attack_loader, P_attack, correct_key, 'final')
    
    print(f"\nFinal Results:")
    print(f"  Mean NTGE: {ntge}")
    print(f"  Median NTGE: {extrapolated_ntge}")
    print(f"  Final GE: {final_ge}")
    print(f"Model saved to: {OUTPUT_DIR}/best_ntge_model.pth")
    print(f"Preprocessing params saved to: {OUTPUT_DIR}/preprocessing_params.npy")
    
    return model


if __name__ == "__main__":
    model = main()
