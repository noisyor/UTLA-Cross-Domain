# Unsupervised Transfer Learning Attack (UTLA)
This repository contains code and data to replicate the results for the paper titled "Attack from Shadows: Unsupervised Side-channel Transfer Learning across Devices and Modalities," to appear in the 21st ACM ASIA Conference on Computer and Communications Security (AsiaCCS 2026). 

## About

In this work, we focus on unsupervised transfer learning attacks in side-channel analysis, where a passive adversary leverages only unlabeled traces from a target device to recover its secret key. This setting reflects a realistic assumption in practical scenarios where collecting labeled traces from the target device is infeasible, particularly for proprietary devices or cloud computing environments. Prior unsupervised deep learning side-channel analysis (DL-SCA) methods assume that a single domain-invariant model suffices across both source and target domains, but this assumption breaks down as device discrepancy increases, such as across different hardware platforms, side-channel modalities, or implementations. We address this gap with the Unsupervised Transfer Learning Attack (UTLA), which uses a separate encoder with a shared classifier to yield lower DL-SCA loss and improved key extraction fidelity. UTLA keeps the source classifier fixed while training a target-specific encoder, using Maximum Mean Discrepancy (MMD) regularization to align feature distributions. Unlike prior approaches that fail under significant domain mismatch, UTLA enables reliable key recovery across diverse scenarios: (a) different hardware platforms, from a Spartan-6 FPGA to an XMEGA MCU using only 61 traces; (b) different leakage modalities, from power measurements on an XMEGA MCU to cache-timing traces on an x86 processor using 435 traces; and (c) different implementations, from masked and shuffled AES on STM32 (ASCADv2) to masked AES on AVR (ASCADv1) using 1519 traces. Our results reveal a significant security implication: public side-channel datasets can serve as effective attack vectors against unseen devices implementing the same cryptographic algorithm.

## Repository Organization
This repository is organized into folders for each of the source device types, including:
1. [XMEGA-Power](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Devices/XMEGA)
2. [XMEGA-EM](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Probe_Positions)
3. [Sakura-AES](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Devices/SAKURA_AES)
4. [ASCADv1(variable-key)](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key)
5. [ASCADv2](https://github.com/ANSSI-FR/ASCAD/tree/master/STM32_AES_v2)
6. [x86-Cache](https://github.com/Cross-Platform-Cache-Attack/x86-high-freq-cache-timing-leakage)
7. [CHES-2025](https://pace-tl.gitbook.io/ches-challenge-2025)

For each source dataset, we train a profiling model on the labeled traces and save it in the ./models subfolder within each folder. The Python code for each attack follows the structure of library imports, function definitions, model architecture definition, model instantiation, training, and testing. We evaluate Guessing Entropy (GE) up to q = 25000 traces and $N_{GE}$ (number of traces to obtain GE = 1) for each of the attacks on the unlabelled target traces. The GE(q) vs q, where q is the number of traces saved in ./figures/ sub-folder, and their corresponding .npy files are saved in the ./results/ folder. The contents of each of the source folders include the following files:
1. Readme file: A description of the specifics of trace and labeled data, including the trace dimensions, dataset size, and chosen window for profiling/transfer learning.
2. Source_profiling.py: Python code from training the profiled model.
3. UTLA_from_X_to_Y.py: UTLA code from source dataset X to target dataset Y, which does not involve finetuning the source model.
