"""Unit tests for data splits, model forward/loss, holdout guards, and inference checks."""

import numpy as np
import pytest
import torch

from src.ml.data import assert_no_leakage, ptbxl_split
from src.ml.evaluate import (
    apply_thresholds,
    guard_test_split,
    tune_class_thresholds,
    validate_checkpoint,
)
from src.ml.inference import prepare
from src.ml.models import ECGNet, WeightedFocalLoss
from src.ml.train import class_weights


def test_patient_leakage_assertion():
    folds = np.array([1, 2, 9, 10])
    patients = np.array([101, 102, 103, 104])
    split = ptbxl_split(folds)
    assert_no_leakage(split, patients)

    leaked_patients = np.array([101, 102, 101, 104])
    with pytest.raises(AssertionError, match="patient in both train and val"):
        assert_no_leakage(split, leaked_patients)


def test_holdout_guard_and_checkpoint_validation():
    with pytest.raises(RuntimeError, match="Holdout guard"):
        guard_test_split("test", name="mitbih", allow_final_test=False)

    with pytest.raises(ValueError, match="missing keys"):
        validate_checkpoint({"state_dict": {}})


def test_threshold_tuning_improves_minority_recall():
    y_true = np.array([0, 0, 0, 1, 1])
    proba = np.array(
        [
            [0.80, 0.20],
            [0.75, 0.25],
            [0.70, 0.30],
            [0.60, 0.40],
            [0.55, 0.45],
        ],
        dtype=np.float32,
    )
    thresholds = tune_class_thresholds(y_true, proba)
    preds = apply_thresholds(proba, thresholds)
    assert (preds == y_true).all()


def test_class_weights_handles_missing_class_without_nan():
    y_partial = np.array([0, 0, 1, 2])
    w = class_weights(y_partial, n_classes=5, power=0.5)
    assert torch.isfinite(w).all()


def test_ecgnet_forward_and_focal_loss():
    model = ECGNet(in_channels=2, n_classes=5, n_extra=5)
    x = torch.randn(4, 2, 252)
    extra = torch.randn(4, 5)
    targets = torch.tensor([0, 1, 2, 3])

    logits = model(x, extra)
    assert logits.shape == (4, 5)

    loss_fn = WeightedFocalLoss(gamma=1.5)
    loss = loss_fn(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)


def test_inference_prepare_rejects_flatline_and_nan():
    flat = np.zeros((1000, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="flatlined"):
        prepare(flat, fs=360)

    bad = np.ones((1000, 2), dtype=np.float64)
    bad[10, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        prepare(bad, fs=360)