import torch.nn as nn


class PeakFinder(nn.Module):
    def __init__(self, channels=8, kernel_size=3):
        super(PeakFinder, self).__init__()

        layers = [
            nn.Conv3d(1, channels, kernel_size=(kernel_size, ) * 3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(kernel_size, ) * 3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(kernel_size,) * 3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, 2, kernel_size=(1, ) * 3)
        ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def num_params(self):
        count = 0
        for param in self.parameters():
            count += param.view(-1).size()[0]
        return count
