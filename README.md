# ECG-APP — Deep Learning Framework for ECG-Based Arrhythmia Classification

University research project with two separate PyTorch1D-CNN tasks:

- PTB-XL named rhythms: SR/AFIB/STACH/SBRAD,12leads,10s,100Hz.
- MIT-BIH AAMI-style heartbeat groups: N/S/V/F/Q,two-channel252-sample windows plus6RR-context features.

Data engineering, preprocessing, controlled training and the final held-out evaluations are complete. **This AI workstream does not implement Streamlit. The separately advanced `origin/main` already contains Streamlit and its own `src/ml/` pipeline; their compatibility with these checkpoints has NOT been implemented or validated.** Anna's integration contract is in [STREAMLIT_INTEGRATION_HANDOFF.md](STREAMLIT_INTEGRATION_HANDOFF.md). Current results are in [EXPERIMENTS_OVERVIEW.md](EXPERIMENTS_OVERVIEW.md) and [docs/experiments](docs/experiments). Earlier phase reports are historical snapshots and may describe installation or setup problems that were later resolved.

## Local environment

Use Linux/Python3.12 and an isolated environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements_preprocessing.txt
.venv/bin/python -m pip install -r requirements_model_cpu.txt
.venv/bin/python -m pip install pytest
```

The model requirements use the official PyTorch CPU wheel index. The previously verified environment used PyTorch2.15.0+cpu. No GPU is assumed. Installing dependencies does not download ECG datasets or checkpoints.

The clone contains source, tests, configurations and aggregate technical reports. It intentionally omits raw/processed ECG, waveform/NPZ files, original clinical metadata, per-record manifests/predictions, virtualenvs, caches and all model weights. Obtain required checkpoint files separately with SHA256 verification; see the handoff. Existing loaders need the documented local export manifests and datasets; they do not create missing exports magically. Do not reprocess/retrain merely because files are absent from a clone.

Public PTB-XL1.0.3 metadata is required locally for metadata-dependent workflows but not redistributed here. If needed, obtain only `ptbxl_database.csv` and `scp_statements.csv` from the official `https://physionet.org/files/ptb-xl/1.0.3/` into `data/ptbxl/metadata/`. This instruction is not a waveform download. The full original database CSV and detailed manifests existed in the initial local Git history; the publication plan removes their tracking in the new tree without deleting local copies or rewriting history. Their removal does not erase old commits.

## Code map

- `src/data_loader.py`, `preprocessing.py`, `segmentation.py`, `label_mapping.py`: MIT-BIH preprocessing.
- `src/prepare_dataset.py`, `class_imbalance.py`: original exports/analysis; do not rerun on frozen exports.
- `src/ptbxl_metadata.py`, `ptbxl_rhythm.py`, `ptbxl_bulk.py`: separate PTB-XL adapters/export infrastructure.
- `src/models/`: three independent model definitions; selected models are PTB-XL CNN and MIT-BIH CNN+RR.
- `src/*_torch.py`, `*_protocol.py`, `*_training.py`: loaders, frozen split/fold protocols and training infrastructure.
- `examples/`: explicit phase scripts. Historical evaluation/training scripts are not application startup code.
- `configs/`: model/training configuration and frozen MIT-BIH patient split. Technical benchmark groups are not actual personal identities.

## Safe lightweight checks

```bash
MPLCONFIGDIR=/tmp/ecg-checks .venv/bin/python -m pytest \
  -p examples.pytest_mitbih_holdout_guard \
  tests/test_ptbxl_metrics.py tests/test_mitbih_f_decisions.py
```

These use synthetic metric inputs, not ECG inference. Do not blindly run legacy all-record QC after a final-test lock. Some integration tests require local datasets; a code-only checkout is not claimed to pass all data-dependent tests. Both final test sets have already been evaluated and must not be reopened for model selection or UI tuning. Phase26 has a global one-attempt claim; do not remove/bypass it. Lock/audit artifacts remain local and must accompany their local results when transferred.

## Collaboration and publication

Working branch: `feature/ptbxl-1dcnn`. Do not force-push or merge remote teammate changes automatically. `docs/PUBLICATION_MANIFEST.json` is an explicit safe-file allowlist with hashes; `scripts/publish_verified_handoff.py` reviews it and can publish only after checking remote divergence. Publication status must be verified separately; creating these documents does not imply a push succeeded.

No Streamlit integration with these AI checkpoints, arbitrary lead adaptation, validated peak detector for this contract, clinical calibration or diagnostic certification is claimed. See [the feature publication plan](docs/FEATURE_BRANCH_PUBLICATION_PLAN.md) for the existing main workstream and manual PR integration requirements. This is a research prototype, not a clinical diagnostic device.
