"""Training loop for MIT-BIH beat and PTB-XL rhythm models.

References
----------
.. [1] Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019).
       Class-balanced loss based on effective number of samples.
       Proceedings of the IEEE/CVF CVPR, 9268-9277.
       Access: https://arxiv.org/abs/1901.05555
.. [2] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
       Focal loss for dense object detection. Proceedings of the IEEE ICCV, 2980-2988.
       Access: https://arxiv.org/abs/1708.02002

       Regarding the first article, I also looked and took inspiration from the publication's github repo:
       https://github.com/richardaecn/class-balanced-loss which has a nice implementation of the
       class-balanced loss, but I ended up using the focal loss instead, which is simpler and works well
       for this task I think.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.ml.data import (MITBIH_CLASSES, PTBXL_CLASSES, ROOT, assert_no_leakage,
                         load_mitbih, load_ptbxl, mitbih_group_split, ptbxl_split,
                         rr_features)
from src.ml.models import ECGNet, count_parameters

MODELS_DIR = ROOT / "models"


def get_data(dataset, rr=False):
    # returns X, extra features, y, the split indices and the class names
    # extra has 0 columns when it is not used, so the rest of the code stays the same
    if dataset == "mitbih":
        d = load_mitbih()
        split = mitbih_group_split(d["y"], d["groups"])
        assert_no_leakage(split, d["groups"], "patient group")
        if rr:
            extra = rr_features(d["positions"], d["record_ids"])
        else:
            extra = np.zeros((len(d["y"]), 0), dtype=np.float32)
        return d["X"], extra, d["y"], split, MITBIH_CLASSES
    d = load_ptbxl()
    split = ptbxl_split(d["strat_fold"])
    assert_no_leakage(split, d["patient_id"], "patient")
    return d["X"], np.zeros((len(d["y"]), 0), dtype=np.float32), d["y"], split, PTBXL_CLASSES


def class_weights(y_train, n_classes, power):
    # computed on the training part only. power=1 is the fully balanced weight,
    # 0.5 is softer (the rarest class would get a huge weight otherwise)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    w = (len(y_train) / (n_classes * counts)) ** power
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []

    with torch.set_grad_enabled(training):
        for xb, eb, yb in loader:
            xb, eb, yb = xb.to(device), eb.to(device), yb.to(device)
            out = model(xb, eb)
            loss = loss_fn(out, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            preds.append(out.argmax(1).cpu().numpy())
            targets.append(yb.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    n = len(targets)
    return {"loss": total_loss / n,
            "acc": float((preds == targets).mean()),
            "f1": float(f1_score(targets, preds, average="macro", zero_division=0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mitbih", "ptbxl"], required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-power", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rr", action="store_true", help="MIT-BIH only: add RR timing features")
    ap.add_argument("--limit", type=int, default=0, help="use only this many training samples (quick check)")
    ap.add_argument("--name", default=None, help="name of the saved model, default is the dataset name")
    args = ap.parse_args()

    # fixed seeds, results can still differ a bit between machines (GPU)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.name or args.dataset

    if args.rr and args.dataset != "mitbih":
        raise SystemExit("--rr only makes sense for mitbih")
    X, extra, y, split, classes = get_data(args.dataset, rr=args.rr)
    n_classes = len(classes)
    train_idx = split["train"]
    if args.limit:
        train_idx = np.random.default_rng(args.seed).permutation(train_idx)[:args.limit]

    # standardise the extra features with the TRAINING part only
    n_extra = extra.shape[1]
    if n_extra:
        extra_mean = extra[split["train"]].mean(0)
        extra_std = extra[split["train"]].std(0) + 1e-6
        extra = ((extra - extra_mean) / extra_std).astype(np.float32)
    else:
        extra_mean = extra_std = np.zeros(0, dtype=np.float32)

    def make_loader(idx, shuffle):
        ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(extra[idx]),
                           torch.from_numpy(y[idx]))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(split["val"], False)

    model = ECGNet(in_channels=X.shape[1], n_classes=n_classes, n_extra=n_extra).to(device)
    weights = class_weights(y[train_idx], n_classes, args.weight_power)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"dataset={args.dataset} device={device} params={count_parameters(model)}")
    print("train/val samples:", len(train_idx), len(split["val"]))
    print("class weights:", dict(zip(classes, weights.numpy().round(2).tolist())))

    MODELS_DIR.mkdir(exist_ok=True)
    best_f1, best_epoch, history = -1.0, 0, []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, loss_fn, device, optimizer)
        va = run_epoch(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(f"epoch {epoch:3d} | train loss {tr['loss']:.4f} f1 {tr['f1']:.3f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.3f} f1 {va['f1']:.3f} | {time.time() - t0:.0f}s")

        # keep the checkpoint with the best validation macro F1
        if va["f1"] > best_f1:
            best_f1, best_epoch = va["f1"], epoch
            torch.save({"state_dict": model.state_dict(),
                        "in_channels": X.shape[1], "n_classes": n_classes,
                        "n_extra": n_extra, "extra_mean": extra_mean.tolist(),
                        "extra_std": extra_std.tolist(),
                        "class_names": list(classes), "epoch": epoch,
                        "val_f1": best_f1, "args": vars(args)},
                       MODELS_DIR / f"{name}.pt")
        elif epoch - best_epoch >= args.patience:
            print(f"no improvement for {args.patience} epochs, stopping")
            break

    (MODELS_DIR / f"{name}_history.json").write_text(json.dumps(history, indent=1))
    print(f"best val macro F1 {best_f1:.4f} at epoch {best_epoch}, saved to models/{name}.pt")


if __name__ == "__main__":
    main()