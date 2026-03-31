Here are the files for Sakura-AES dataset.

Step-1: Extract the zip files in the data folder (after downloading from the this [link](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Devices/SAKURA_AES/Data)).

Step-2: Run profiling step using Source_Profiling.py
Set train_first_time to 1 for training or to 0 for inference on source dataset
Choose source_device from {1, 2, 3, 4, 5, 6, 7, 8}
```
python Source_Profiling.py train_first_time source_device
```

Step-4: Run transfer attacks.

For Sakura-AES to XMEGA-EM UTLA:
```
python UTLA_from_Sakura_to_XMEGA-EM.py set_UTLA_train source_device target_device
```

For Sakura-AES to XMEGA-Power UTLA:
```
python UTLA_from_Sakura_to_XMEGA-Power.py set_UTLA_train source_device target_device
```
