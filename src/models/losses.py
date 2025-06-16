
import torch.nn.functional as F
import torch.nn as nn
import torch

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)

        # BCE
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # Dice
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return bce + (1 - dice)