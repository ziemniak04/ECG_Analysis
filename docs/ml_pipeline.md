# ML pipeline: technical documentation

Notes for the modelling part of the project (1D-CNN models, training, metrics,
Streamlit app). The preprocessing and the data export are described in
`README_preprocessing.md` and `ECG_DATA_HANDOFF_FOR_ANNA.md` in the repo root.
This file only covers what was added on top of that.

Status: data loading and splits are done. Models, training, metrics and the app
are still to do (there is a checklist at the bottom).

## 1. Setup (Windows, PowerShell)

Python 3.12 is used (tested with 3.12.10).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_phase4.txt
python -m pip install torch scikit-learn streamlit plotly tqdm
```

All commands below are run from the repo root with the venv active, and with
`python -m ...` (not by running the file directly), otherwise the `src` imports
do not work.

## 2. Getting the data

The `data/` folder is not in git, so every new machine has to build it.

**MIT-BIH.** The documented way (`wfdb.dl_database`) did not work here: the
download stopped with `RemoteDisconnected` (PhysioNet closed the connection,
wfdb downloads many files in parallel). So the whole database was downloaded as
one zip file with PowerShell instead.

```powershell

# download (about 75 MB, no progress bar shown for visual simplicity)
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip" -OutFile mitdb.zip

# unzip and move the folder to the path that the loader expects
Expand-Archive mitdb.zip -DestinationPath mitdb_unzipped
Move-Item mitdb_unzipped\mit-bih-arrhythmia-database-1.0.0 data\mitdb\physionet.org\files\mitdb\1.0.0
```

The `Move-Item` target must not exist yet, otherwise the folder ends up nested one
level deeper. Check the files:

```powershell
(dir data\mitdb\physionet.org\files\mitdb\1.0.0\*.hea).Count   # the answer should be 48
(dir data\mitdb\physionet.org\files\mitdb\1.0.0\*.dat).Count   # here 48
(dir data\mitdb\physionet.org\files\mitdb\1.0.0\*.atr).Count   # and here 49 (102-0.atr is the extra one)
```

Then delete the leftovers and build the dataset:

```powershell
Remove-Item -Recurse -Force mitdb.zip, mitdb_unzipped
python -m src.prepare_dataset
```

The script prints one line per record and a summary at the end. Expected
summary: 109452 beats, 48 records, 47 patient groups, classes N=90594, S=2781,
V=7235, F=802, Q=8040, and `source_files_verified: true`. The output goes to
`data/processed/ecg_dataset_v1`. The script refuses to write into a folder that
already exists, so to rebuild, delete that folder first.

If the zip download fails too, download it in the browser from
https://physionet.org/content/mitdb/1.0.0/ and continue from the `Expand-Archive`
step.

**PTB-XL.**

```powershell
python -m src.ptbxl_bulk                            # dry run, no download
python -m src.ptbxl_bulk --execute --approve-bulk   # about 470 MB
```

Expected result: `status: complete`, 19605 successful, 0 failed. If it stops
because of the network, running the same command again continues where it
stopped. Output goes to `data/processed/ptbxl_rhythm_v1`.

Be patient, because it does take time. It heavily depends on the machine, but expect 1 hour up to a couple of hours of downloading.

## 3. Loading the data and the splits (`src/ml/data.py`)

```powershell
python -m src.ml.data
```

This loads both datasets, builds the splits, checks that no patient is in two
parts, and prints a table with the class counts per part.

The first run is slow because it reads every shard (PTB-XL is one file per ECG).
It then saves two cache files in `data/processed/cache/`:

- `mitbih_v1.npz`
- `ptbxl_v1.npz`

Later runs load the cache. If the exported data is ever rebuilt, delete the
cache folder, otherwise the old arrays get loaded. `load_mitbih(use_cache=False)`
also forces a reload.

### Array shapes

| dataset | X | y | extra |
|---|---|---|---|
| MIT-BIH | (109452, 2, 252) float32 | 0..4 = N, S, V, F, Q | `groups`, `record_ids` |
| PTB-XL | (19605, 12, 1000) float32 | 0..3 = SR, AFIB, STACH, SBRAD | `patient_id`, `strat_fold`, `ecg_id` |

MIT-BIH windows are single beats at 360 Hz. PTB-XL signals are 10 s at 100 Hz.
The two datasets are never mixed, they are different tasks.

### Split protocol

Rule for everything: the split is by patient, never by beat or by ECG.

**MIT-BIH.** 70/15/15 by patient group (47 groups). The groups are shuffled with
a fixed seed (`seed=42`) until every part has at least 20 beats of every class,
because S and F beats are concentrated in a few patients. Records 201 and 202
belong to the same patient and share one group.

**PTB-XL.** The official `strat_fold` column: folds 1-8 train, fold 9 validation,
fold 10 test. This is the split recommended by the dataset authors and it already
keeps patients apart. `assert_no_leakage` checks the patient ids as well.

### Reproducibility

The MIT-BIH split is deterministic. Running the code again gives exactly the same
split as long as the data and these parameters stay the same: `seed`,
`val_frac`, `test_frac`, `min_per_class`. If you wish to change the training, or
just run it after the training, then do not change them after training has
started, otherwise results from different runs cannot be compared. The PTB-XL
split has no randomness at all.

Current split (output of `python -m src.ml.data`):

MIT-BIH

| part | n | N | S | V | F | Q |
|---|---|---|---|---|---|---|
| train | 72839 | 62837 | 1953 | 3783 | 395 | 3871 |
| val | 17322 | 12934 | 101 | 1822 | 374 | 2091 |
| test | 19291 | 14823 | 727 | 1630 | 33 | 2078 |

PTB-XL

| part | n | SR | AFIB | STACH | SBRAD |
|---|---|---|---|---|---|
| train | 15689 | 13354 | 1189 | 643 | 503 |
| val | 1958 | 1667 | 146 | 82 | 63 |
| test | 1958 | 1666 | 149 | 80 | 63 |

## 4. Known issues and decisions

### MIT-BIH: very few F beats in test and S beats in validation

Because the split is by patient and the rare classes sit in a few patients, the
parts are uneven:

- the test set has only **33 F beats**
- the validation set has only **101 S beats**

Consequences:

- F-class recall and F1 on the test set will be very noisy, a few beats change
  the number a lot. This must be mentioned wherever test results are reported.
- Early stopping / model selection on validation is also less reliable for S.

Options that were considered:

1. Keep this split and report the limitation. (current choice)
2. Raise `min_per_class` (for example to 100) before any training. This gives a
   more balanced split but changes the split, so it only makes sense before the
   first model is trained.
3. Use `mitbih_cv_folds` (5-fold, patient groups kept apart) and report the mean
   and spread over folds. More stable, but 5 times more training.

To see which records the 33 F beats come from:

```powershell
python -c "from src.ml.data import *; import numpy as np; m=load_mitbih(); s=mitbih_group_split(m['y'],m['groups']); i=s['test']; r=m['record_ids'][i]; f=m['y'][i]==3; print({x:int((r[f]==x).sum()) for x in np.unique(r[f])})"
```

Result: (i will fill in after running it)

### Class imbalance

Both datasets are heavily imbalanced (MIT-BIH: N is about 83 percent, F under
1 percent; PTB-XL: SR is about 85 percent). Class weights or sampling must be
then computed from the training part only. Validation and test are never
rebalanced. Because of the imbalance, accuracy alone is not a good metric, so
macro F1, per-class recall and AUPRC are reported next to it.

### Other limits worth mentioning

- The two MIT-BIH channels are not always the same lead (for example record 114
  is V5/MLII, some records use V2 or V4 instead of V1).
- Preprocessing was done offline on whole recordings (zero-phase filter,
  per-recording z-score), so it is not causal.
- No 50 Hz notch filter was used. At 100 Hz sampling it would sit at the Nyquist
  frequency, and for MIT-BIH the residual powerline noise was measured and found
  small.
- Beat windows come from the expert annotations. For an uploaded file the app has
  no annotations, so it needs an R-peak detector.

## 5. Commit style

Conventional Commits: `type(scope): short description`, for example
`feat(data): add loaders and patient-separated splits`.
Types that will probably be used (i will edit this later): `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `build`.
Scopes: `data`, `model`, `train`, `metrics`, `app`.

## 6. Checklist

- [x] MIT-BIH and PTB-XL data built and checked
- [x] data loading, patient-separated splits, leakage check (`src/ml/data.py`)
- [ ] 1D-CNN for MIT-BIH (5 classes)
- [ ] 1D-CNN for PTB-XL (4 classes)
- [ ] training and validation loop
- [ ] metrics: accuracy, recall, specificity, macro F1, AUPRC, inference time
- [ ] Streamlit app
- [ ] `requirements_ml.txt` with pinned versions
- [ ] add `models/` to `.gitignore`