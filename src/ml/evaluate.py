"""Evaluate a trained model on the validation or held-out test split.

References
----------
.. [1] Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more
       informative than the ROC plot when evaluating binary classifiers on
       imbalanced datasets. PLoS ONE, 10(3), e0118432.
       Access: https://pmc.ncbi.nlm.nih.gov/articles/PMC4349800/
.. [2] Lipton, Z. C., Elkan, C., & Naryanaswamy, B. (2014). Optimal thresholding of
       classifiers to maximize F1 measure. ECML PKDD, 225-239.
       Access: https://arxiv.org/abs/1402.1892
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix

from src.ml.models import ECGNet
from src.ml.train import MODELS_DIR, get_data

RESULTS_DIR = MODELS_DIR.parent / "results" / "ml"


def predict_proba(model, X, extra, device, batch_size=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            eb = torch.from_numpy(extra[i:i + batch_size]).to(device)
            out.append(torch.softmax(model(xb, eb), dim=1).cpu().numpy())
    return np.concatenate(out)


def inference_time(model, X, extra, device, n=200, warmup=20):
    # time for ONE sample at a time (like the app will do it)
    model.eval()
    times = []
    with torch.no_grad():
        for i in range(warmup + n):
            x = torch.from_numpy(X[i % len(X)][None]).to(device)
            e = torch.from_numpy(extra[i % len(X)][None]).to(device)
            t0 = time.perf_counter()
            model(x, e)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.median(times))


def compute_metrics(y_true, proba, class_names):
    k = len(class_names)
    pred = proba.argmax(1)
    cm = confusion_matrix(y_true, pred, labels=range(k))

    per_class = {}
    for c, name in enumerate(class_names):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        # one-vs-rest AUPRC, undefined if the class is not in this split
        is_c = (y_true == c)
        auprc = average_precision_score(is_c, proba[:, c]) if is_c.any() else float("nan")
        per_class[name] = {"support": int(cm[c].sum()), "precision": float(precision),
                           "recall": float(recall), "specificity": float(specificity),
                           "f1": float(f1), "auprc": float(auprc)}

    def macro(key):
        return float(np.nanmean([v[key] for v in per_class.values()]))

    summary = {"accuracy": float((pred == y_true).mean()),
               "macro_recall": macro("recall"), "macro_specificity": macro("specificity"),
               "macro_f1": macro("f1"), "macro_auprc": macro("auprc")}
    return summary, per_class, cm


def plot_confusion(cm, class_names, title, path):
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="share of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mitbih", "ptbxl"], required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--name", default=None, help="model name in models/, default is the dataset name")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    name = args.name or args.dataset
    device = torch.device(args.device)
    ckpt = torch.load(MODELS_DIR / f"{name}.pt", map_location=device)

    # old checkpoints have no n_extra, they were trained without RR features
    n_extra = ckpt.get("n_extra", 0)
    X, extra, y, split, classes = get_data(args.dataset, rr=n_extra > 0)
    if list(classes) != ckpt["class_names"]:
        raise ValueError("class names in the checkpoint do not match the dataset")
    if n_extra:
        # same standardisation as in training (numbers saved in the checkpoint)
        extra = (extra - np.array(ckpt["extra_mean"])) / np.array(ckpt["extra_std"])
        extra = extra.astype(np.float32)
    idx = split[args.split]
    X, extra, y = X[idx], extra[idx], y[idx]

    model = ECGNet(ckpt["in_channels"], ckpt["n_classes"], n_extra=n_extra).to(device)
    model.load_state_dict(ckpt["state_dict"])

    proba = predict_proba(model, X, extra, device)
    summary, per_class, cm = compute_metrics(y, proba, classes)
    t_mean, t_median = inference_time(model, X, extra, device)

    print(f"\n{name} on {args.split} ({len(y)} samples), checkpoint from epoch {ckpt['epoch']}")
    print(f"{'class':<7}{'n':>7}{'prec':>8}{'recall':>8}{'spec':>8}{'f1':>8}{'auprc':>8}")
    for c, m in per_class.items():
        print(f"{c:<7}{m['support']:>7}{m['precision']:>8.3f}{m['recall']:>8.3f}"
              f"{m['specificity']:>8.3f}{m['f1']:>8.3f}{m['auprc']:>8.3f}")
    print("\nconfusion matrix (rows = true, columns = predicted):")
    print(cm)
    print("\n" + "  ".join(f"{k}={v:.3f}" for k, v in summary.items()))
    print(f"inference time, batch size 1 on {device}: mean {t_mean:.2f} ms, median {t_median:.2f} ms")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{args.split}"
    result = {"model": name, "split": args.split, "n_samples": int(len(y)),
              "checkpoint_epoch": ckpt["epoch"], "summary": summary, "per_class": per_class,
              "confusion_matrix": cm.tolist(), "class_names": list(classes),
              "inference_ms_mean": t_mean, "inference_ms_median": t_median,
              "device": str(device)}
    (RESULTS_DIR / f"{stem}_metrics.json").write_text(json.dumps(result, indent=1))
    plot_confusion(cm, classes, f"{name} ({args.split})", RESULTS_DIR / f"{stem}_confusion.png")
    print(f"saved to results/ml/{stem}_metrics.json and {stem}_confusion.png")


if __name__ == "__main__":
    main()