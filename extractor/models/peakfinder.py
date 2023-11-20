import torch.nn as nn


class PeakFinder(nn.Module):
    def __init__(self, channels_input=1, channels_output=2):
        super(PeakFinder, self).__init__()

        layers = [
            nn.Conv3d(channels_input, 8, kernel_size=(5, ) * 3, padding=2),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 8, kernel_size=(3, ) * 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 4, kernel_size=(3, ) * 3, padding=1),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),
            nn.Conv3d(4, channels_output, kernel_size=(1, ) * 3)
        ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def num_params(self):
        count = 0
        for param in self.parameters():
            count += param.view(-1).size()[0]
        return count
