"""1D Residual CNN architecture and imbalance-aware loss for ECG classification.

Used for both beat-level classification:
    (MIT-BIH: in_channels=2, n_classes=5,length=252, optional n_extra=5 RR features)
    and rhythm-level classification (PTB-XL: in_channels=12, n_classes=4, length=1000)

References
----------
.. [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for
       image recognition. Proceedings of the IEEE CVPR, 770-778.
       Access: https://arxiv.org/abs/1512.03385
.. [2] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
       Focal loss for dense object detection. Proceedings of the IEEE ICCV, 2980-2988.
       Access: https://arxiv.org/abs/1708.02002

    I just want to mention that the first article even though is about image recognition,
    but the architecture is very general and can be applied to 1D signals as well, and that
    is where I took inspiration from.

    The second article was also not about ECG classification, but it was a very good article
    that explained the focal loss and how it can be used to deal with class imbalance, which
    seems to be a common problem not only in ECG classification, but universally as well.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    """Residual 1D convolutional block with BatchNorm and LeakyReLU."""

    def __init__(self, in_ch, out_ch, stride=1, kernel=7):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.1)

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.skip(x))


class ECGNet(nn.Module):
    """1D Residual CNN with AdaptiveAvgPool1d and optional RR-interval fusion"""

    def __init__(self, in_channels, n_classes, widths=(32, 64, 128), dropout=0.4, n_extra=0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, widths[0], 7, padding=3, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.LeakyReLU(0.1),
        )

        blocks = []
        prev = widths[0]
        for i, w in enumerate(widths):
            blocks.append(ResBlock(prev, w, stride=1 if i == 0 else 2))
            prev = w
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(prev + n_extra, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, extra=None):
        x = self.blocks(self.stem(x))
        x = self.pool(x).squeeze(-1)
        if extra is not None and extra.shape[1] > 0:
            x = torch.cat([x, extra], dim=1)
        return self.head(x)


class WeightedFocalLoss(nn.Module):
    """focal loss with optional class weights to focus training on hard minority beats"""

    def __init__(self, weight=None, gamma=1.5):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.weight is not None:
            loss = self.weight[targets] * loss
        return loss.mean()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)