# Unsupervised Transfer Learning Attack (UTLA)
This repository contains code and data to replicate the results of the manuscript submission to CCS 2025 Cycle B (Submission Number: 1390) titled "Attack from Shadows: Unsupervised Side-channel Transfer across Devices and Modalities." 

## Repository Organization
This repository is organized into folder for each of the source device type, including:
1. [XMEGA-Power](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Devices/XMEGA)
2. [XMEGA-EM](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Probe_Positions)
3. [Sakura-AES](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Devices/SAKURA_AES)
4. [ASCADv1(variable-key)](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key)
5. [ASCADv2](https://github.com/ANSSI-FR/ASCAD/tree/master/STM32_AES_v2)
6. [x86-Cache](https://github.com/Cross-Platform-Cache-Attack/x86-high-freq-cache-timing-leakage)

For each of the source datasets, we train a profiling model using the labelled traces which is saved in the ./models sub-folder inside each of the folders. The python code for each of the attack follow the structure of library imports, function definitions, model architecture definition, model instantiation, training, and testing. We evaluate Guessing Entropy (GE) upto q = 5000 traces and $N_{GE}$ (number of traces to obtain GE = 1) for each of the attacks on the unlabelled target traces. The GE(q) vs q, where q are the number of traces are saved in ./figures/ sub-folder and there corresponding .npy files are saved in the ./results/ folder. The contents of each of source folder includes the following files:
1. Readme file: A description on the specifics of trace and labelled data including the trace dimensions, dataset size, and chosen window for profiling/transfer learning.
2. Source_profiling.py: Python code from training the profiled model.
3. Direct_Transfer_from_X_to_Y.py: Direct Transfer (DT) codes from source dataset X to target dataset Y which does nto involve finetuning the source model.
4. UTLA_from_X_to_Y.py: UTLA code from source dataset X to target dataset Y which does nto involve finetuning the source model.
