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


class LassoLoss(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, ground, model):
        # cross_entropy = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        loss = mse(pred, ground)
        for layer in model.layers:
            params = torch.cat([x.view(-1) for x in layer.parameters()])
            loss += self.alpha * torch.norm(params, 1)  # L1 regularisation term
        return loss
