Download dataset.zip -> dataset.h5 from this [repository](https://github.com/Cross-Platform-Cache-Attack/x86-high-freq-cache-timing-leakage).

Here are the files for x86-Cache dataset.

Step-1: Run profiling step using Source_Profiling.py
Set train_first_time to 1 for training or to 0 for inference on source dataset
```
python Source_Profiling.py train_first_time
```

Step-2: Run direct transfer or x86-to-XMEGA transfer attacks.

```
python x86_Direct_Transfer.py train_first_time source_device target_device print_intermediate_GE
python x86_Direct_Transfer_GELoss.py train_first_time source_device target_device print_intermediate_GE
python x86_XMEGA_cross_UTLA.py set_UTLA_train source_device target_device
python x86_XMEGA_cross_UTLA_autoenc_initial.py set_UTLA_train source_device target_device auto_enc_train
```
