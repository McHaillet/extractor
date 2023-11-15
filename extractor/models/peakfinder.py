import torch.nn as nn


class PeakFinder(nn.Module):
    def __init__(self, channels_input=2, channels_first=8, channels_second=4, channels_output=2, kernel_size=3):
        super(PeakFinder, self).__init__()

        layers = [
            nn.Conv3d(channels_input, channels_first, kernel_size=(kernel_size, ) * 3, padding=1),
            nn.BatchNorm3d(channels_first),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels_first, channels_second, kernel_size=(kernel_size, ) * 3, padding=1),
            nn.BatchNorm3d(channels_second),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels_second, channels_output, kernel_size=(1, ) * 3)
        ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def num_params(self):
        count = 0
        for param in self.parameters():
            count += param.view(-1).size()[0]
        return count
