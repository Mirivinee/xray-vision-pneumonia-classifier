"""Baseline scratch CNN for NORMAL vs PNEUMONIA chest X-ray classification.

Input : float32 tensor [batch, 3, 224, 224], values in [0, 1] (from data_pipeline.py).
Output: raw logits, shape [batch]. Apply sigmoid for P(PNEUMONIA).
        Labels follow data_pipeline.CLASS_TO_LABEL: NORMAL=0, PNEUMONIA=1.

Architecture: 4 x (Conv3x3 -> BatchNorm -> ReLU -> MaxPool2), global average
pooling, dropout, one linear unit. Spatial size: 224 -> 112 -> 56 -> 28 -> 14.
"""

import torch
from torch import nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class ScratchCNN(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), dropout=0.5, in_channels=3):
        super().__init__()
        blocks = []
        previous = in_channels
        for width in channels:
            blocks.append(conv_block(previous, width))
            previous = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(previous, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)

    @property
    def gradcam_target_layer(self):
        """Last conv layer, for Vineet's Grad-CAM."""
        return self.features[-1][0]


if __name__ == "__main__":
    model = ScratchCNN()
    out = model(torch.rand(4, 3, 224, 224))
    params = sum(p.numel() for p in model.parameters())
    print(f"output shape: {tuple(out.shape)}, parameters: {params:,}")