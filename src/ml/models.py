"""1D-CNN with residual blocks, used for both datasets.

MIT-BIH: in_channels=2,  n_classes=5  (input length 252, optionally n_extra=5 RR features)
PTB-XL:  in_channels=12, n_classes=4  (input length 1000)
"""
import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel=7):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.1)

        # the skip connection needs a 1x1 conv when the shape changes
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
            # first block keeps the length, the others halve it
            blocks.append(ResBlock(prev, w, stride=1 if i == 0 else 2))
            prev = w
        self.blocks = nn.Sequential(*blocks)

        # global average pooling, so the head does not care about the input length
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(prev + n_extra, 64),  # n_extra = extra features next to the CNN output
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, extra=None):
        x = self.blocks(self.stem(x))
        x = self.pool(x).squeeze(-1)
        if extra is not None and extra.shape[1] > 0:
            x = torch.cat([x, extra], dim=1)
        return self.head(x)  # raw logits, softmax is applied outside


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)