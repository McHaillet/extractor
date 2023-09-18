import torch
import torch.nn as nn
import torch.nn.functional as F


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts Nx1xDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 5D input image (Nx1xDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    new_shape = list(input.size())
    new_shape[1] = C

    # scatter to get the one-hot tensor
    return torch.zeros(new_shape).to(input.device).scatter_(1, input, 1)


class TverskyLoss(nn.Module):
    def __init__(self, classes: int, alpha: float = 0.5, beta: float = 0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.cl = classes
        self.eps = 1e-7

    def forward(self, y_pred, y_true):

        y_true_onehot = expand_as_one_hot(y_true, C=self.cl).float()
        prob = F.softmax(y_pred, dim=1)

        dims = (0, 2, 3, 4)
        intersection = torch.sum(prob * y_true_onehot, dims)
        fps = torch.sum(prob * (1 - y_true_onehot), dims)
        fns = torch.sum((1 - prob) * y_true_onehot, dims)

        num = intersection
        den = intersection + (self.alpha * fps) + (self.beta * fns)
        tversky_index = (num / (den + self.eps))

        return self.cl - tversky_index.sum()
