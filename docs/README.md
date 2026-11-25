# Documentation

This codebase builds an end-to-end ECG arrhythmia classification pipeline that turns raw MIT-BIH and PTB-XL recordings into patient-separated datasets, trains 1D Residual CNNs for beat and rhythm classification, and serves real-time predictions in a Streamlit viewer.

| File | Contents |
|---|---|
| [docs/ml_pipeline.md](ml_pipeline.md) | Setup, data building, data loading, patient-separated splits, known issues, progress checklist for the ML part |

Documentation for the signal preprocessing, dataset audits, and export pipeline lives in the repository root:

* `README_preprocessing.md`
* `DATASET_FOR_ANNA.md`
* `ECG_DATA_HANDOFF_FOR_ANNA.md`
* `FINAL_ECG_DATA_VALIDATION_REPORT.md`
* `FINAL_ECG_VISUAL_VALIDATION.md`
* `FINAL_MITBIH_EXPORT_REPORT.md`
* `PHASE13_PTBXL_PREPROCESSING_PLAN.md`
* `PHASE14_FULL_PTBXL_EXPORT_PLAN.md`
* `PROPOSED_RHYTHM_PREPROCESSING_PLAN.md`
* `PTBXL_DATASET_FOR_ANNA.md`
* `PTBXL_FULL_EXPORT_REPORT.md`
* `PTBXL_METADATA_AUDIT.md`
* `PTBXL_PREPROCESSING_PILOT_REPORT.md`
* `PTBXL_RHYTHM_LABEL_PROPOSAL.md`
* `RHYTHM_CLASS_FEASIBILITY.md`
* `RHYTHM_DATASET_AUDIT.md`