Download `ascad-variable.h5` from [here](https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-083349/ascad-variable.h5). The scripts expect this file in the current folder with the name `ascad-variable.h5`.
Download the ascadv1.h5 from [here](https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip). 
For more details check [here](https://github.com/ANSSI-FR/ASCAD/tree/master).

Here are the files for ASCADv1-variable-key dataset.

Step-1: Run profiling step using Source_Profiling.py.

```
python Source_Profiling.py train_first_time print_intermediate_GE
```

Step-2: Run direct transfer attacks.

```
python ASCAD_Direct_Transfer_ASCADv1.py train_first_time print_intermediate_GE
python ASCAD_Direct_Transfer_ASCADv1_usingGE.py train_first_time print_intermediate_GE
```

Step-3: Run UTLA transfers.

For ASCADv1 variable-key to XMEGA-EM:

```
python ASCADV1_XMEGA_EM_Transfer_UTLA.py set_UTLA_train source_device target_device
```

For ASCADv1 variable-key to ASCAD fixed-key/desynchronized traces:

```
python ASCADV1_to_ASCAD_Transfer_UTLA.py set_UTLA_train source_device target_device
```
