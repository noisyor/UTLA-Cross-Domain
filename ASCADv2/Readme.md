Download the ASCADv2 dataset from [here](https://www.data.gouv.fr/fr/datasets/ascadv2/#/resources/a6cf925c-079c-4468-a723-d94bce6c31f8). The scripts expect the extracted HDF5 file in the current folder with the name `ascadv2.h5`.

Here are the files for ASCADv2 dataset.

Step-1: Run profiling step using Source_Profiling.py
Set train_first_time to 1 for training or to 0 for inference on source dataset
```
python Source_Profiling.py train_first_time
```

For the profiling test script:

```
python Source_Profiling_test.py train_first_time print_intermediate_GE
```

Step-2: Run direct transfer attack.

```
python ASCADV2_Direct_Transfer.py train_first_time print_intermediate_GE
```

Step-3: Run ASCADv2 to ASCADv1 variable-key UTLA.

```
python ASCADV2_to_ASCADV1_Transfer_UTLA.py set_UTLA_train source_device target_device print_intermediate_GE auto_enc_train
```

This script expects the ASCADv1 variable-key dataset and pre-trained ASCADv1 model in `../ASCADv1-variable-key/`.
