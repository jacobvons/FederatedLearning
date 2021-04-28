import torch
import torch.nn as nn


class RidgeLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, ground, model):
        mse = nn.MSELoss()
        loss = mse(pred, ground)
        for layer in model.layers:
            loss += torch.sum(torch.square(layer.weight))
        return loss


class MSELoss(nn.MSELoss):

    def __init__(self):
        super().__init__()

    def forward(self, input, target, model):
        return super().forward(input, target)
