# ML pipeline: technical documentation

Notes for the modelling part of the project (1D-CNN models, training, metrics,
Streamlit app). The preprocessing and the data export are described in
`README_preprocessing.md` and `ECG_DATA_HANDOFF_FOR_ANNA.md` in the repo root.
This file only covers what was added on top of that.

Status: data loading, splits, models, training and evaluation on the validation
split are done. The final test evaluation and the Streamlit app are still to do
(there is a checklist at the bottom).

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
# create the parent folder, Move-Item below needs it
New-Item -ItemType Directory -Force data\mitdb\physionet.org\files\mitdb | Out-Null

# download (about 75 MB, no progress bar shown for visual simplicity)
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip" -OutFile mitdb.zip

# unzip and move the folder to the path that the loader expects
Expand-Archive mitdb.zip -DestinationPath mitdb_unzipped
Move-Item mitdb_unzipped\mit-bih-arrhythmia-database-1.0.0 data\mitdb\physionet.org\files\mitdb\1.0.0
```

If an earlier failed download left a `data\mitdb` folder, delete it first
(`Remove-Item -Recurse -Force data\mitdb`). The `Move-Item` target
(`...\1.0.0`) must not exist yet, otherwise the folder ends up nested one
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

- `mitbih_v2.npz` (also holds the R-peak positions, needed for the RR features)
- `ptbxl_v1.npz`

Later runs load the cache. If the exported data is ever rebuilt, delete the
cache folder, otherwise the old arrays get loaded. Deleting the cache folder is
always safe, it is only a speed-up. `load_mitbih(use_cache=False)` also forces a
reload. An old `mitbih_v1.npz` is no longer used and can be deleted.

### Array shapes

| dataset | X | y | extra |
|---|---|---|---|
| MIT-BIH | (109452, 2, 252) float32 | 0..4 = N, S, V, F, Q | `groups`, `record_ids`, `positions` |
| PTB-XL | (19605, 12, 1000) float32 | 0..3 = SR, AFIB, STACH, SBRAD | `patient_id`, `strat_fold`, `ecg_id` |

MIT-BIH windows are single beats at 360 Hz (0.25 s before and 0.45 s after the
R-peak). PTB-XL signals are 10 s at 100 Hz. The two datasets are never mixed,
they are different tasks.

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
started, otherwise results from different runs cannot be compared. The `--seed`
of the training script only changes the model initialisation and batch order,
not the split. The PTB-XL split has no randomness at all.

Current split (output of `python -m src.ml.data`, identical after the cache was
rebuilt for the RR features):

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

## 4. Models and training

Files: `src/ml/models.py` (model), `src/ml/train.py` (training loop).

### Architecture (`ECGNet`)

One 1D-CNN for both datasets: a stem convolution (kernel 7), three residual
blocks (32, 64, 128 channels, kernel 7, stride 1/2/2, BatchNorm, LeakyReLU),
global average pooling, then a head Linear(64) - LeakyReLU - Dropout(0.4) -
Linear(classes). It returns logits, softmax is applied outside.

| model | input | classes | parameters |
|---|---|---|---|
| MIT-BIH baseline | 2 x 252 | 5 | 249,989 |
| MIT-BIH + RR features | 2 x 252 + 5 features | 5 | 250,309 |
| PTB-XL | 12 x 1000 | 4 | 252,164 |

### Training setup

Weighted cross-entropy, weights = (n_train / (k * count_c)) ** 0.5 computed on
the training part only (power 0.5 = softer than fully balanced, the rarest class
would get a huge weight otherwise). Adam, lr 1e-3, weight decay 1e-4, batch size
256, at most 40 epochs, seed 42. The best epoch is chosen by validation macro F1,
training stops after 8 epochs without improvement. The test split is never used
in training. About 90-110 s per epoch for MIT-BIH and about 70 s for PTB-XL on
CPU (no GPU is used, the CPU is sufficient for this model size).

### RR timing features (MIT-BIH only, `--rr`)

A single beat window cannot show whether a beat came early. For every beat, 5
features are computed from the R-peak positions of the neighbouring beats in the
same recording: pre-RR and post-RR interval (s), mean of the last 10 intervals,
and the two ratios to that mean. The first/last beat of a recording reuse the
neighbouring interval. The features are standardised with the training part only
(mean and std are saved in the checkpoint). Nothing comes from other patients,
so there is no leakage across the split. The app will get the positions from an
R-peak detector.

### Commands

```powershell
python -m src.ml.train --dataset mitbih --name mitbih            # baseline
python -m src.ml.train --dataset mitbih --rr --name mitbih_rr    # with RR features
python -m src.ml.train --dataset ptbxl                           # name defaults to ptbxl
python -m src.ml.train --dataset mitbih --limit 3000 --epochs 2 --name quick_check   # quick test
```

Checkpoints go to `models/<name>.pt` (gitignored) together with a history file.

## 5. Evaluation (`src/ml/evaluate.py`)

```powershell
python -m src.ml.evaluate --dataset mitbih --name mitbih_rr --split val
python -m src.ml.evaluate --dataset ptbxl --split val
```

Reports accuracy, per-class precision / recall / specificity / F1 / AUPRC, the
macro averages (plain mean over classes), the confusion matrix, and the
inference time for one sample at a time on CPU. Specificity and AUPRC are
one-vs-rest per class (AUPRC from the softmax probabilities). Results are saved
in `results/ml/` as JSON and a confusion matrix PNG.

Rule: `--split val` for every decision. `--split test` is run once, at the end,
and nothing is changed afterwards.

## 6. Results on the validation split

Models after early stopping (best epoch): MIT-BIH baseline epoch 9 (of 17),
MIT-BIH + RR epoch 13 (of 21), PTB-XL epoch 7 (of 15).

### MIT-BIH baseline (17322 beats)

| class | n | precision | recall | specificity | F1 | AUPRC |
|---|---|---|---|---|---|---|
| N | 12934 | 0.966 | 0.923 | 0.906 | 0.944 | 0.982 |
| S | 101 | 0.000 | 0.000 | 0.992 | 0.000 | 0.016 |
| V | 1822 | 0.763 | 0.920 | 0.966 | 0.834 | 0.909 |
| F | 374 | 0.276 | 0.377 | 0.978 | 0.319 | 0.221 |
| Q | 2091 | 0.979 | 0.994 | 0.997 | 0.986 | 0.998 |

### MIT-BIH + RR features (17322 beats)

| class | n | precision | recall | specificity | F1 | AUPRC |
|---|---|---|---|---|---|---|
| N | 12934 | 0.968 | 0.898 | 0.911 | 0.932 | 0.980 |
| S | 101 | 0.026 | 0.059 | 0.987 | 0.036 | 0.038 |
| V | 1822 | 0.829 | 0.945 | 0.977 | 0.883 | 0.951 |
| F | 374 | 0.283 | 0.684 | 0.962 | 0.401 | 0.396 |
| Q | 2091 | 0.879 | 0.885 | 0.983 | 0.882 | 0.940 |

### PTB-XL (1958 ECGs)

| class | n | precision | recall | specificity | F1 | AUPRC |
|---|---|---|---|---|---|---|
| SR | 1667 | 0.979 | 0.970 | 0.880 | 0.974 | 0.997 |
| AFIB | 146 | 0.883 | 0.932 | 0.990 | 0.907 | 0.919 |
| STACH | 82 | 0.918 | 0.817 | 0.997 | 0.865 | 0.928 |
| SBRAD | 63 | 0.582 | 0.730 | 0.983 | 0.648 | 0.750 |

### Summary

| model | accuracy | macro recall | macro specificity | macro F1 | macro AUPRC | inference (ms, CPU, 1 sample) |
|---|---|---|---|---|---|---|
| MIT-BIH baseline | 0.914 | 0.643 | 0.968 | 0.617 | 0.625 | 1.17 |
| MIT-BIH + RR | 0.892 | 0.694 | 0.964 | 0.627 | 0.661 | 1.20 |
| PTB-XL | 0.953 | 0.862 | 0.962 | 0.848 | 0.898 | 3.21 |

### Reading the results

- PTB-XL works well. The weakest class is SBRAD (mostly confused with SR, which
  is only a heart-rate threshold).
- MIT-BIH: training macro F1 reaches about 0.93-0.95 while validation stays at
  0.45-0.63 and jumps between epochs. The model does not transfer well to new
  patients. This is the expected problem of a patient-wise split on MIT-BIH.
- The RR features were expected to help class S (early beats). They did not,
  S stays close to zero (6 of 101 correct). V and F improved on validation, Q got
  worse. Macro F1 is 0.617 vs 0.627, which is a tie.
- These validation differences cannot be trusted much, see the next section: the
  validation score of the rare classes depends on one or two patients each.

## 7. Known issues and decisions

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
   and spread over folds. More stable, but 5 times more training. Optional, only
   if time allows (an overnight job on CPU).

For now, the decision is to keep the current split (option 1). It is due to a fact that F is concentrated in very few patients, so a stricter split would remove training examples.

To see which records the 33 F beats come from:

```powershell
python -c "from src.ml.data import *; import numpy as np; m=load_mitbih(); s=mitbih_group_split(m['y'],m['groups']); i=s['test']; r=m['record_ids'][i]; f=m['y'][i]==3; print({x:int((r[f]==x).sum()) for x in np.unique(r[f])})"
```

Result: records 223 (14 beats), 233 (11), 124 (5), 201 (2) and 202 (1), so 5 records
(read from the output of `split_report`, see below).

### MIT-BIH: very few patients behind the rare classes

`python -m src.ml.split_report` lists which records hold the beats of each class
in each split (records with the most beats, beats in brackets):

| class | train | validation | test |
|---|---|---|---|
| S | 232 (1382 of 1953), 222 (209), 207 (107) | **118 (96 of 101)** | 209 (383), 201 (128), 223 (73), 202 (55), 234 (50), 124 (31) |
| F | **213 (362 of 395)**, 205 (11), 210 (10) | **208 (372 of 374)** | 223 (14), 233 (11), 124 (5), 201 (2), 202 (1) |
| Q | 104 (2062), 217 (1802) | **102 (2083 of 2091)** | **107 (2078 of 2078)** |
| V | 200 (826), 106 (520), 119 (444) | 208 (992), 203 (444), 214 (256) | 233 (830), 223 (473), 201 (198) |

Split sizes: 33 records in train, 7 in validation, 8 in test.

Consequences:

- Validation S is one patient (118), validation F is one patient (208),
  validation Q is one patient (102). The validation macro F1 therefore mostly
  measures how three individual patients are handled. This explains why it jumps
  from epoch to epoch, and why the choice of the "best epoch" is partly luck.
- The comparison baseline vs RR on validation is the same: V and F improved
  mainly because of record 208, Q got worse because of record 102. It is not
  evidence about the classes in general.
- Class F in training is mostly one patient (213). Class Q comes from the four
  paced-rhythm records of the database (102, 104, 107, 217), so "Q" is mostly
  paced beats.
- The test split is better spread for S (6+ records) and V, but F has only 33
  beats from 5 records and Q is a single record (107). F and Q test results must
  be reported with this caveat.
- TODO: I have to check the PhysioNet notes for record 118 (S beats there are, 
  as far as I know, aberrated beats in a bundle branch block
  record, which would explain why they are not recognised as S).
- This is a property of MIT-BIH (47 patients), not a bug: no patient-wise split
  can give every rare class many patients in every part.

### Model choice for MIT-BIH

Decided before looking at test results: both models (baseline and RR) are
evaluated once on the test split and both are reported as a comparison. The app
uses the RR model (`mitbih_rr`), because it is better on V (PVC, one of the
arrhythmias of the project) and F and has the higher macro AUPRC, with the
caveat above that the evidence is weak. The R-peak detector needed for the app
gives the positions for the RR features as well.

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
  no annotations, so it needs an R-peak detector (its positions also feed the RR
  features).
- Each model was trained once (seed 42), the variation between seeds was not
  measured.

## 8. Full pipeline (order of commands)

```powershell
# setup (once)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_phase4.txt
python -m pip install torch scikit-learn streamlit plotly tqdm

# data (once, MIT-BIH via the zip as described in section 2)
python -m src.prepare_dataset
python -m src.ptbxl_bulk --execute --approve-bulk

# loading, splits, who is in which split
python -m src.ml.data
python -m src.ml.split_report

# training
python -m src.ml.train --dataset mitbih --name mitbih
python -m src.ml.train --dataset mitbih --rr --name mitbih_rr
python -m src.ml.train --dataset ptbxl

# validation (decisions)
python -m src.ml.evaluate --dataset mitbih --name mitbih --split val
python -m src.ml.evaluate --dataset mitbih --name mitbih_rr --split val
python -m src.ml.evaluate --dataset ptbxl --split val

# test, ONCE, after all decisions
python -m src.ml.evaluate --dataset mitbih --name mitbih --split test
python -m src.ml.evaluate --dataset mitbih --name mitbih_rr --split test
python -m src.ml.evaluate --dataset ptbxl --split test
```

## 9. Commit style

Conventional Commits: `type(scope): short description`, for example
`feat(data): add loaders and patient-separated splits`.
Types that will probably be used (i will edit this later): `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `build`.
Scopes: `data`, `model`, `train`, `metrics`, `app`.

## 10. Checklist

- [x] MIT-BIH and PTB-XL data built and checked
- [x] data loading and patient-separated splits (`src/ml/data.py`)
- [x] 1D-CNN models for MIT-BIH (5 classes) and PTB-XL (4 classes)
- [x] training and validation loop
- [x] evaluation: accuracy, recall, specificity, macro F1, AUPRC, inference time (validation)
- [ ] evaluation on the test split (once)
- [ ] Streamlit app