import torch
from torch import nn as nn


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    assert input.dim() == 4

    shape = input.size()
    shape = list(shape)
    shape.insert(1, C)
    shape = tuple(shape)

    # expand the input tensor to Nx1xDxHxW
    src = input.unsqueeze(1)

    if ignore_index is not None:
        # create ignore_index mask for the result
        expanded_src = src.expand(shape)
        mask = expanded_src == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        src = src.clone()
        src[src == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, src, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, src, 1)


class TverskyLoss(nn.Module):
    def __init__(self, classes: int, alpha: float = 0.5, beta: float = 0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.cl = classes
        self.eps = 1e-7

    def forward(self, y_pred, y_true):

        y_true_onehot = expand_as_one_hot(y_true, C=self.cl).float()
        prob = y_pred  # F.softmax(y_pred, dim=1)

        dims = (0, 2, 3, 4)
        intersection = torch.sum(prob * y_true_onehot, dims)
        fps = torch.sum(prob * (1 - y_true_onehot), dims)
        fns = torch.sum((1 - prob) * y_true_onehot, dims)

        num = intersection
        den = intersection + (self.alpha * fps) + (self.beta * fns)
        tversky_index = (num / (den + self.eps))

        return self.cl - tversky_index.sum()
