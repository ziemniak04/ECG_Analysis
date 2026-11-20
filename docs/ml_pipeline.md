# ML pipeline: technical documentation

Notes for the modelling part of the project (1D-CNN models, training, metrics,
Streamlit app). The preprocessing and the data export are described in
`README_preprocessing.md` and `ECG_DATA_HANDOFF_FOR_ANNA.md` in the repo root.
This file only covers what was added on top of that.

Status: data loading, splits, models, training, evaluation (validation and test)
and the Streamlit app are done (there is a checklist at the bottom).

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
- These validation differences cannot be trusted much, see section 8: the
  validation score of the rare classes depends on one or two patients each.

## 7. Results on the test split

Run once, after all decisions were made (the three commands of section 10). Nothing
was changed afterwards.

### MIT-BIH baseline (19291 beats)

| class | n | precision | recall | specificity | F1 | AUPRC |
|---|---|---|---|---|---|---|
| N | 14823 | 0.912 | 0.800 | 0.743 | 0.852 | 0.924 |
| S | 727 | 0.355 | 0.135 | 0.990 | 0.195 | 0.179 |
| V | 1630 | 0.666 | 0.782 | 0.964 | 0.719 | 0.873 |
| F | 33 | 0.000 | 0.030 | 0.885 | 0.001 | 0.002 |
| Q | 2078 | 0.985 | 0.889 | 0.998 | 0.934 | 0.989 |

### MIT-BIH + RR features (19291 beats)

| class | n | precision | recall | specificity | F1 | AUPRC |
|---|---|---|---|---|---|---|
| N | 14823 | 0.830 | 0.812 | 0.448 | 0.821 | 0.878 |
| S | 727 | 0.625 | 0.371 | 0.991 | 0.466 | 0.434 |
| V | 1630 | 0.949 | 0.925 | 0.995 | 0.937 | 0.977 |
| F | 33 | 0.002 | 0.121 | 0.898 | 0.004 | 0.002 |
| Q | 2078 | 0.098 | 0.038 | 0.958 | 0.055 | 0.265 |

### PTB-XL (1958 ECGs)

| class | n | precision | recall | specificity | F1 | AUPRC |
|---|---|---|---|---|---|---|
| SR | 1666 | 0.969 | 0.964 | 0.822 | 0.966 | 0.993 |
| AFIB | 149 | 0.871 | 0.906 | 0.989 | 0.888 | 0.908 |
| STACH | 80 | 0.887 | 0.787 | 0.996 | 0.834 | 0.881 |
| SBRAD | 63 | 0.486 | 0.571 | 0.980 | 0.526 | 0.605 |

### Validation vs test

| model | split | accuracy | macro recall | macro specificity | macro F1 | macro AUPRC | inference (ms, CPU, 1 sample) |
|---|---|---|---|---|---|---|---|
| MIT-BIH baseline | val | 0.914 | 0.643 | 0.968 | 0.617 | 0.625 | 1.17 |
| MIT-BIH baseline | test | 0.781 | 0.527 | 0.916 | 0.540 | 0.594 | 1.34 |
| MIT-BIH + RR | val | 0.892 | 0.694 | 0.964 | 0.627 | 0.661 | 1.20 |
| MIT-BIH + RR | test | 0.721 | 0.453 | 0.858 | 0.456 | 0.511 | 1.26 |
| PTB-XL | val | 0.953 | 0.862 | 0.962 | 0.848 | 0.898 | 3.21 |
| PTB-XL | test | 0.940 | 0.807 | 0.947 | 0.804 | 0.847 | 3.45 |

### Reading the test results

- **PTB-XL generalises as expected.** Test macro F1 0.804 is close to the
  validation value 0.848. AFIB (F1 0.888) and STACH (0.834) work well, SBRAD is
  the weakest class (F1 0.526, mostly confused with SR in both directions).
- **MIT-BIH validation did not predict test.** Macro F1 dropped from 0.617 to
  0.540 (baseline) and from 0.627 to 0.456 (RR), and the ranking of the two
  models flipped. This is what section 8 warned about: validation S, F and Q
  were one patient each.
- **RR features help the classes that have many patients.** On the test split S is
  spread over 6+ records, and the RR model is clearly better there: S F1 0.195 ->
  0.466 and S AUPRC 0.179 -> 0.434. V improves too: F1 0.719 -> 0.937, AUPRC
  0.873 -> 0.977. This supports the idea that timing (early beats) matters for S,
  which the validation split could not show.
- **RR features hurt Q.** Q F1 drops from 0.934 to 0.055: 1999 of the 2078 Q
  beats (all from record 107) are predicted as N. Q is a single test patient, and
  the likely reason (not checked) is that regular paced beats look normal when
  only the timing is considered. Macro F1 is pulled down mainly by this class.
- **Class F is not recognised by either model** (F1 <= 0.004, 33 beats from 5
  records), and both models call about 2000 normal beats F (2194 baseline, 1932
  RR). F was learned mostly from one training patient (record 213). F predictions
  of the app must not be trusted.
- **Caveats:** one training run per model (seed 42), rare classes depend on very
  few patients, so differences between the two MIT-BIH models should not be
  over-interpreted. The V and S improvements are the most reliable ones because
  they come from many records.

## 8. Known issues and decisions

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

Decided before looking at the test results: both models (baseline and RR) are
evaluated once on the test split and both are reported as a comparison, and the
app uses the RR model (`mitbih_rr`) as default, because it was better on V (PVC,
one of the arrhythmias of the project), F and macro AUPRC on validation.

After the test run (section 7) the picture is mixed: RR is much better for S and
V, but much worse for Q, and worse in macro F1. The pre-declared default was kept
and nothing was retrained or re-selected on the test set. The app lets the user
choose between both models, and the report presents both with the full tables.
Whoever reads the report should know that the RR default was fixed before the
test results, and that on the test split the baseline has the higher macro F1.

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
- In the app the beats come from an automatic R-peak detector (`wfdb` XQRS) and
  not from expert annotations as in training, so the windows can be shifted by a
  few samples. Files with another sampling rate can be resampled, which the
  models never saw in training.

## 9. Streamlit app (`streamlit_app.py`)

```powershell
streamlit run streamlit_app.py
```

Run from the repo root with the venv active, it opens in the browser. It needs the
trained checkpoints in `models/`. Files: `streamlit_app.py` (the page),
`src/ml/inference.py` (reading files, preprocessing, running the models),
`src/ml/viewer.py` (tables and figures) and `.streamlit/config.toml` (colours).
Inference and viewer code contain no streamlit, so they are tested without the app
(`tests/test_inference.py`, `tests/test_viewer.py`). Tested with Streamlit 1.67; it
needs a recent version (`st.fragment`, row selection in tables), if something is
missing run `python -m pip install -U streamlit`.

**Layout.** The sidebar is only for input: upload a WFDB record (`.hea` and `.dat`
together) or a `.csv` (one column per lead, sampling rate typed in), or open a record
from the local `data/` folder, then the model. The task is chosen from the number of
leads: 2 leads are analysed beat by beat (MIT-BIH models), 12 leads in 10 s windows
(PTB-XL model). Under "Advanced": resampling (off by default, the models never saw
resampled data) and the maximum length that is analysed (default 30 min).

The main page, from top to bottom:

1. Summary: number of beats (or windows), heart rate, number and share of flagged ones
   and the count per class.
2. Overview of the whole recording: the filtered signal of the chosen lead for the
   full length, and under it where the flags are (flagged beats per time bin and class,
   or a ribbon with the rhythm of every window). The part that is shown in detail is
   marked in blue.
3. A position slider, "Previous flagged" / "Next flagged", lead and window length.
4. Detail view: raw and filtered signal of the chosen stretch. Beats are dots, flagged
   beats carry their class letter and a light red block behind them. For rhythm,
   every 10 s window is shaded with its class, and "Show all 12 leads" shows all
   leads one under the other.
5. List of everything that was flagged, on the right. A class filter above it, click a
   row to jump to that beat, csv download of the whole list. Normal beats can be
   added to the list with the filter.
6. "About the model": test metrics of the chosen model (read from
   `results/ml/<model>_test_metrics.json`) and its limits.

**Speed.** Analysis (filtering, R-peak detection, classification) runs once per
recording and settings, not on every click. The slider, buttons and list live in a
`st.fragment`, so only that part is redrawn. The overview draws a 30 minute recording
with about 4000 points (min and max per bucket, so peaks stay visible), the detail
view only draws the shown seconds. Measured on a 30 minute, 2 lead recording
(648000 samples): analysis about 3.5 s once, redrawing both figures about 35 ms.

**Preprocessing** in the app is the same code as in training: filter the whole
recording (0.5-45 Hz, order 4, zero-phase, no notch), z-score per lead, then 90
samples before and 162 after each R-peak (beats) or 10 s windows (rhythm).
`tests/test_inference.py` checks that this is identical to `src/preprocessing.py`
and `src/segmentation.py`.

For records from the original datasets the app tells whether the record was in the
training, validation or test split (needs the cache from `python -m src.ml.data`),
because results on a training record are better than on a new patient.

Limits: flagged means "not N" (beats) or "not SR" (rhythm), there are no clinical
rules on top and no episode detection (a run of V beats is a list of single beats).
Class F predictions are not reliable (section 7). It is a prototype and not a
medical device.

## 10. Full pipeline (order of commands)

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

# app
streamlit run streamlit_app.py
```

## 11. Commit style

Conventional Commits: `type(scope): short description`, for example
`feat(data): add loaders and patient-separated splits`.
Types that will probably be used (i will edit this later): `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `build`.
Scopes: `data`, `model`, `train`, `metrics`, `app`.

## 12. Checklist

- [x] MIT-BIH and PTB-XL data built and checked
- [x] data loading and patient-separated splits (`src/ml/data.py`)
- [x] 1D-CNN models for MIT-BIH (5 classes) and PTB-XL (4 classes)
- [x] training and validation loop
- [x] evaluation: accuracy, recall, specificity, macro F1, AUPRC, inference time
- [x] evaluation on the test split (run once)
- [x] Streamlit app