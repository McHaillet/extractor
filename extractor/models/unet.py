import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


# Reference U-Nets
# Adapted from https://github.com/cosmic-cortex/pytorch-UNet

# 2D reference U-Net implementation
class UNet2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, filters=(64, 128, 256, 512, 1024)):
        assert len(filters) > 2, 'filters must have at least 3 members'

        super(UNet2D, self).__init__()

        if isinstance(in_channels, list):
            in_channels = in_channels[0]
        if isinstance(out_channels, list):
            out_channels = out_channels[0]

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First2D(in_channels, filters[0], filters[0]))
        encoder_layers.extend([Encoder2D(filters[i], filters[i + 1], filters[i + 1])
                               for i in range(len(filters) - 2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder2D(2 * filters[i + 1], 2 * filters[i], 2 * filters[i], filters[i])
                               for i in reversed(range(len(filters) - 2))])
        decoder_layers.append(Last2D(filters[1], filters[0], out_channels[0] if isinstance(out_channels, list) else out_channels))

        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center2D(filters[-2], filters[-1], filters[-1], filters[-2])
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x, return_all=False):
        x_enc = [x]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))

        x_dec = [self.center(x_enc[-1])]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1 - dec_layer_idx]
            x_cat = torch.cat(
                [pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))

        if not return_all:
            return x_dec[-1]
        else:
            return x_enc + x_dec

    def num_params(self):
        count = 0
        for param in self.parameters():
            count += param.view(-1).size()[0]
        return count


# 3D reference U-Net implementation
class UNet3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 filters: tuple[int, ...] = (16, 32, 64),  # reduced params, before was: (64, 128, 256, 512, 1024)
                 dropout: Union[bool, float] = False):
        assert len(filters) > 2, 'filters must have at least 3 members'

        super(UNet3D, self).__init__()

        if isinstance(in_channels, list):
            in_channels = in_channels[0]
        if isinstance(out_channels, list):
            out_channels = out_channels[0]

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First3D(in_channels, filters[0], filters[0], dropout=dropout))
        encoder_layers.extend([Encoder3D(filters[i], filters[i + 1], filters[i + 1], dropout=dropout)
                               for i in range(len(filters)-2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder3D(2 * filters[i + 1], 2 * filters[i], 2 * filters[i], filters[i],
                                         dropout=dropout)
                               for i in reversed(range(len(filters)-2))])
        decoder_layers.append(Last3D(filters[1], filters[0], out_channels))

        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center3D(filters[-2], filters[-1], filters[-1], filters[-2], dropout=dropout)
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x, return_all=False):
        x_enc = [x]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))

        x_dec = [self.center(x_enc[-1])]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1-dec_layer_idx]
            x_cat = torch.cat(
                [pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))

        if not return_all:
            return x_dec[-1]
        else:
            return x_enc + x_dec

    def num_params(self):
        count = 0
        for param in self.parameters():
            count += param.view(-1).size()[0]
        return count


# Utilities
def pad_to_shape(this, shp):
    """
    Pads this image with zeroes to shp.
    Args:
        this: image tensor to pad
        shp: desired output shape

    Returns:
        Zero-padded tensor of shape shp.
    """
    if len(shp) == 4:
        pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    elif len(shp) == 5:
        pad = (0, shp[4] - this.shape[4], 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    return F.pad(this, pad)


# 2D building blocks
class First2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class Encoder2D(nn.Module):
    def __init__(
            self, in_channels, middle_channels, out_channels,
            dropout=False, downsample_kernel=2
    ):
        super(Encoder2D, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=downsample_kernel),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Center2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center2D, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.center = nn.Sequential(*layers)

    def forward(self, x):
        return self.center(x)


class Decoder2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Last2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Last2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1)
        ]

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


# 3D building blocks
class First3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class Encoder3D(nn.Module):
    def __init__(
            self, in_channels, middle_channels, out_channels,
            dropout=False, downsample_kernel=2
    ):
        super(Encoder3D, self).__init__()

        layers = [
            nn.MaxPool3d(kernel_size=downsample_kernel),
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Center3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center3D, self).__init__()

        layers = [
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.center = nn.Sequential(*layers)

    def forward(self, x):
        return self.center(x)


class Decoder3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Last3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Last3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=1)
        ]

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)
