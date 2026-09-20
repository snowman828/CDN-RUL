# Data

The datasets are **not** redistributed here. Both are publicly available from
the NASA Prognostics Center of Excellence Data Repository:

| Dataset | Used for | Source |
|---|---|---|
| C-MAPSS (FD001-FD004) | Tables 1-3, 5, 6; Figures 1-4 | https://data.nasa.gov (Turbofan Engine Degradation Simulation Data Set) |
| N-CMAPSS (DS02-006) | Tables 1, 3, 4; Figure 5; Section 4.6/4.7 | https://ti.arc.nasa.gov (N-CMAPSS) |

Place the extracted files so that `src/common.py` finds them:

```
data/raw/
  train_FD001.txt  test_FD001.txt  RUL_FD001.txt
  train_FD002.txt  test_FD002.txt  RUL_FD002.txt
  train_FD003.txt  test_FD003.txt  RUL_FD003.txt
  train_FD004.txt  test_FD004.txt  RUL_FD004.txt
  N-CMAPSS_DS02-006.h5
```

`src/ncmapss_data.py` streams the HDF5 file with per-unit subsampling
(sampling=10, i.e. 0.1 Hz) so the full file never has to be held in memory.
