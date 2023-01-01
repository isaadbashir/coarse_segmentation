import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def CE_loss(input_logits, target_targets, ignore_index = 255, temperature=1):
    return F.cross_entropy(input_logits / temperature, target_targets, ignore_index=ignore_index)




class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):

        ce = F.cross_entropy(input, target)
        smooth = 1e-5
        _, input = torch.max(input, 1)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.contiguous().view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * ce + dice