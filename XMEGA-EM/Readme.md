Here are the files for XMEGA-EM dataset.

Step-1: Extract the zip files in the data folder (after downloading from this [link](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Probe_Positions/Data)).

Step-2: Run Generate_ID_model.sh to generate identity labels (skip this step if identity labels are already there)

Step-3: Run profiling step using Source_Profiling.py.

Step-4: Run transfer attacks.

For in-domain XMEGA-EM to XMEGA-EM UTLA:
```
python UTLA_from_XMEGA-EM_to_XMEGA-EM.py set_UTLA_train source_device target_device
```

For XMEGA-EM to XMEGA-Power direct transfer:
```
python Direct_Transfer_from_XMEGA-EM_to_XMEGA-Power.py source_device target_device
```

For XMEGA-EM to XMEGA-Power UTLA:
```
python UTLA_from_XMEGA-EM_to_XMEGA-Power.py set_UTLA_train source_device target_device
```
