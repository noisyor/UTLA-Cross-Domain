Here are the files for XMEGA-Power dataset.

Step-1: Extract the zip files in the data folder (after downloading from [here](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Devices/XMEGA/Data)).

Step-2: Run Generate_ID_model.sh to generate identity labels (skip this step if identity labels are already there).

Step-3: Run profiling step using Source_Profiling.py
Set train_first_time to 1 for training or to 0 for inference on source dataset
Choose source_device from {1, 2, 3, 4, 5, 6, 7, 8}
```
python Source_Profiling.py train_first_time source_device
```

Step-4: Run the attack.
