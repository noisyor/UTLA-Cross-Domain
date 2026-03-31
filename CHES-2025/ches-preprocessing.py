#!/usr/bin/env python
"""
Preprocess CHES Challenge traces and save to HDF5.

Preprocessing pipeline (matches train_ches_cnn.py EXACTLY):
1. Load FULL traces (7000 samples)
2. PCA denoising on FULL traces (project to PCA space and back, keeping 95% variance)
3. Global standardization on FULL traces (zero mean, unit variance per sample point)
4. Extract window [TRACE_OFFSET : TRACE_OFFSET + TRACE_LENGTH] for saving

NOTE: The denoising and standardization are computed on full traces to match
      train_ches_cnn.py behavior, then the window is extracted for saving.

Usage:
    python ches-preprocessing.py
"""

import os
import sys
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.decomposition import PCA

# ============================================================================
# Configuration
# ============================================================================

# Input path. Override with environment variables when data is stored elsewhere.
CHES_RAW_PATH = os.environ.get("CHES_RAW_PATH", "./CHES_Challenge.h5")

# Output path
CHES_PREPROCESSED_PATH = os.environ.get(
    "CHES_PREPROCESSED_PATH",
    "./CHES_Challenge_preprocessed.h5",
)

# Trace window parameters (extracted AFTER preprocessing on full traces)
TRACE_OFFSET = 1200
TRACE_LENGTH = 500

# PCA denoising parameters
USE_DENOISING = True
PCA_VARIANCE_RATIO = 0.95     # Keep 95% variance
MAX_TRACES_FOR_PCA = 50000    # Use subset for PCA fitting (memory)

# Dataset sizes (set to None to process all)
TRAIN_NUM = None  # Process all training traces
ATTACK_NUM = None  # Process all attack traces


def compute_denoising_matrix(X_profiling, n_components=0.95):
    """
    Compute denoising matrix using PCA on profiling traces.
    
    Denoising is done by projecting to PCA space and back:
    X_denoised = X @ V.T @ V
    where V are the principal components.
    
    Args:
        X_profiling: FULL profiling traces (n_traces, n_samples) - typically 7000 samples
        n_components: Variance ratio (0-1) to keep
    
    Returns:
        denoising_matrix: Matrix to project traces for denoising (n_samples, n_samples)
        pca: Fitted PCA object (for metadata)
    """
    print("Computing denoising matrix...")
    
    # Use a subset for PCA if dataset is very large
    if len(X_profiling) > MAX_TRACES_FOR_PCA:
        print(f"  Using subset of {MAX_TRACES_FOR_PCA} traces for PCA fitting")
        indices = np.random.choice(len(X_profiling), MAX_TRACES_FOR_PCA, replace=False)
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
    
    return denoising_matrix.astype(np.float32), pca


def denoise_traces(X, denoising_matrix, chunk_size=10000):
    """
    Apply denoising matrix to traces.
    
    Args:
        X: FULL traces (n_traces, n_samples) - typically 7000 samples
        denoising_matrix: PCA denoising matrix (n_samples, n_samples)
        chunk_size: Process in chunks to save memory
    
    Returns:
        X_denoised: Denoised traces
    """
    print("Denoising traces...")
    X_denoised = np.zeros_like(X, dtype=np.float32)
    
    for i in tqdm(range(0, len(X), chunk_size), desc="  Denoising"):
        end = min(i + chunk_size, len(X))
        X_denoised[i:end] = X[i:end].astype(np.float32) @ denoising_matrix
    
    return X_denoised


def compute_standardization_params(X_profiling):
    """
    Compute mean and std from profiling traces for global standardization.
    
    Args:
        X_profiling: FULL profiling traces (n_traces, n_samples)
    
    Returns:
        mean: Per-sample mean (n_samples,)
        std: Per-sample std (n_samples,)
    """
    print("Computing standardization parameters...")
    mean = np.mean(X_profiling, axis=0).astype(np.float32)
    std = np.std(X_profiling, axis=0).astype(np.float32)
    std[std == 0] = 1  # Avoid division by zero
    return mean, std


def standardize_traces(X, mean, std, chunk_size=50000):
    """
    Apply global standardization using precomputed mean/std.
    
    Args:
        X: FULL traces (n_traces, n_samples)
        mean: Per-sample mean
        std: Per-sample std
    
    Returns:
        X_standardized: Standardized traces
    """
    print("Standardizing traces...")
    X_standardized = np.zeros_like(X, dtype=np.float32)
    
    for i in tqdm(range(0, len(X), chunk_size), desc="  Standardizing"):
        end = min(i + chunk_size, len(X))
        X_standardized[i:end] = (X[i:end].astype(np.float32) - mean) / std
    
    return X_standardized


def main():
    print("="*60)
    print("CHES Challenge Trace Preprocessing")
    print("="*60)
    print(f"Input:  {CHES_RAW_PATH}")
    print(f"Output: {CHES_PREPROCESSED_PATH}")
    print(f"Window: offset={TRACE_OFFSET}, length={TRACE_LENGTH} (extracted AFTER preprocessing)")
    print(f"Denoising: {'PCA on FULL traces (variance=%.0f%%)' % (PCA_VARIANCE_RATIO*100) if USE_DENOISING else 'Disabled'}")
    print(f"Normalization: Global standardization on FULL traces")
    print("="*60)
    
    # Check if output already exists
    if os.path.exists(CHES_PREPROCESSED_PATH):
        overwrite = os.environ.get("CHES_OVERWRITE", "").lower() in {"1", "true", "yes", "y"}
        if overwrite or not sys.stdin.isatty():
            print("Output file already exists. Overwriting.")
        else:
            response = input(f"Output file already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load raw data - FULL TRACES
    print("\nLoading raw CHES data (FULL traces)...")
    with h5py.File(CHES_RAW_PATH, 'r') as f:
        print(f"  Available keys: {list(f.keys())}")
        
        # =====================
        # Load training data - FULL TRACES
        # =====================
        print(f"\nLoading profiling/training traces (FULL)...")
        n_train = TRAIN_NUM if TRAIN_NUM else f['Profiling_traces/traces'].shape[0]
        X_train_full = f['Profiling_traces/traces'][:n_train]
        
        # Load metadata
        metadata_train = f['Profiling_traces/metadata'][:n_train]
        Y_train = np.array(metadata_train['labels'])
        P_train = np.array(metadata_train['plaintext'][:, 0])  # First byte
        
        print(f"  Full trace shape: {X_train_full.shape}")  # Should be (N, 7000)
        
        # =====================
        # Load attack data - FULL TRACES
        # =====================
        print(f"\nLoading attack traces (FULL)...")
        n_attack = ATTACK_NUM if ATTACK_NUM else f['Attack_traces/traces'].shape[0]
        X_attack_full = f['Attack_traces/traces'][:n_attack]
        
        metadata_attack = f['Attack_traces/metadata'][:n_attack]
        P_attack = np.array(metadata_attack['plaintext'][:, 0])
        K_attack = np.array(metadata_attack['key'][:, 0])
        correct_key = int(K_attack[0])
        
        print(f"  Full trace shape: {X_attack_full.shape}")  # Should be (N, 7000)
        print(f"  Correct key: {correct_key}")
    
    # =====================
    # Preprocessing on FULL TRACES (matches train_ches_cnn.py)
    # =====================
    print("\n" + "="*60)
    print("Preprocessing on FULL traces...")
    print("="*60)
    
    # Convert to float32
    X_train_processed = X_train_full.astype(np.float32)
    X_attack_processed = X_attack_full.astype(np.float32)
    del X_train_full, X_attack_full
    
    # Step 1: PCA Denoising on FULL traces
    denoising_matrix = None
    n_pca_components = 0
    explained_variance = 0.0
    full_trace_length = X_train_processed.shape[1]
    
    if USE_DENOISING:
        denoising_matrix, pca = compute_denoising_matrix(X_train_processed, n_components=PCA_VARIANCE_RATIO)
        n_pca_components = pca.n_components_
        explained_variance = pca.explained_variance_ratio_.sum()
        
        X_train_processed = denoise_traces(X_train_processed, denoising_matrix)
        X_attack_processed = denoise_traces(X_attack_processed, denoising_matrix)
    else:
        print("Skipping denoising (USE_DENOISING=False)")
    
    # Step 2: Global Standardization on FULL traces
    mean_full, std_full = compute_standardization_params(X_train_processed)
    
    X_train_processed = standardize_traces(X_train_processed, mean_full, std_full)
    X_attack_processed = standardize_traces(X_attack_processed, mean_full, std_full)
    
    # Step 3: Extract window AFTER preprocessing
    print(f"\nExtracting window [{TRACE_OFFSET}:{TRACE_OFFSET + TRACE_LENGTH}]...")
    X_train_window = X_train_processed[:, TRACE_OFFSET:TRACE_OFFSET + TRACE_LENGTH]
    X_attack_window = X_attack_processed[:, TRACE_OFFSET:TRACE_OFFSET + TRACE_LENGTH]
    print(f"  Train window shape: {X_train_window.shape}")
    print(f"  Attack window shape: {X_attack_window.shape}")
    
    # Also extract windowed mean/std for the saved params
    mean_window = mean_full[TRACE_OFFSET:TRACE_OFFSET + TRACE_LENGTH]
    std_window = std_full[TRACE_OFFSET:TRACE_OFFSET + TRACE_LENGTH]
    
    del X_train_processed, X_attack_processed
    
    # =====================
    # Save preprocessed data
    # =====================
    print(f"\nSaving preprocessed data to {CHES_PREPROCESSED_PATH}...")
    with h5py.File(CHES_PREPROCESSED_PATH, 'w') as f:
        # Metadata
        f.attrs['trace_offset'] = TRACE_OFFSET
        f.attrs['trace_length'] = TRACE_LENGTH
        f.attrs['full_trace_length'] = full_trace_length
        f.attrs['preprocessing'] = 'pca_denoising_on_full + global_standardization_on_full + windowing'
        f.attrs['use_denoising'] = USE_DENOISING
        f.attrs['pca_variance_ratio'] = PCA_VARIANCE_RATIO
        f.attrs['pca_n_components'] = n_pca_components
        f.attrs['pca_explained_variance'] = explained_variance
        
        # Training data (windowed)
        train_grp = f.create_group('train')
        train_grp.create_dataset('traces', data=X_train_window, compression='gzip')
        train_grp.create_dataset('labels', data=Y_train)
        train_grp.create_dataset('plaintext', data=P_train)
        
        # Attack data (windowed)
        attack_grp = f.create_group('attack')
        attack_grp.create_dataset('traces', data=X_attack_window, compression='gzip')
        attack_grp.create_dataset('plaintext', data=P_attack)
        attack_grp.create_dataset('correct_key', data=correct_key)
        
        # Preprocessing parameters (windowed for applying to new windowed data)
        # Note: For new data, you'd need to apply full preprocessing then window
        params_grp = f.create_group('preprocessing_params')
        params_grp.create_dataset('mean_window', data=mean_window)
        params_grp.create_dataset('std_window', data=std_window)
        params_grp.create_dataset('mean_full', data=mean_full)
        params_grp.create_dataset('std_full', data=std_full)
        if denoising_matrix is not None:
            params_grp.create_dataset('denoising_matrix', data=denoising_matrix)
    
    # Verify
    print("\nVerifying saved data...")
    with h5py.File(CHES_PREPROCESSED_PATH, 'r') as f:
        print(f"  train/traces: {f['train/traces'].shape}")
        print(f"  train/labels: {f['train/labels'].shape}")
        print(f"  attack/traces: {f['attack/traces'].shape}")
        print(f"  correct_key: {f['attack/correct_key'][()]}")
        print(f"  preprocessing: {f.attrs['preprocessing']}")
        print(f"  full_trace_length: {f.attrs['full_trace_length']}")
        if USE_DENOISING:
            print(f"  PCA components: {f.attrs['pca_n_components']}")
            print(f"  Explained variance: {f.attrs['pca_explained_variance']*100:.1f}%")
    
    # File size
    size_mb = os.path.getsize(CHES_PREPROCESSED_PATH) / (1024 * 1024)
    print(f"\nOutput file size: {size_mb:.1f} MB")
    print("Done!")


if __name__ == '__main__':
    main()
