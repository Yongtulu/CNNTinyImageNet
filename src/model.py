import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, pool=True, dropout_rate=0.1):
        super().__init__()
        layers = []
        for i in range(num_convs):
            in_c = in_channels if i == 0 else out_channels
            layers += [
                nn.Conv2d(in_c, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        layers.append(nn.Dropout2d(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TinyImageNetCNN(nn.Module):
    def __init__(self, num_classes=200, dropout_fc=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   64,  num_convs=2, pool=True,  dropout_rate=0.1),  # 64→32
            ConvBlock(64,  128, num_convs=2, pool=True,  dropout_rate=0.1),  # 32→16
            ConvBlock(128, 256, num_convs=2, pool=True,  dropout_rate=0.2),  # 16→8
            ConvBlock(256, 512, num_convs=2, pool=False, dropout_rate=0.2),  # 8→8
            ConvBlock(512, 512, num_convs=1, pool=False, dropout_rate=0.3),  # 8→8
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(1024, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(num_classes=200, dropout_fc=0.5):
    return TinyImageNetCNN(num_classes=num_classes, dropout_fc=dropout_fc)
