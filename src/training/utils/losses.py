import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def CE_loss(input_logits, target_targets, ignore_index = 255, temperature=1):
    return F.cross_entropy(input_logits / temperature, target_targets, ignore_index=ignore_index)


class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        
        _, predict = torch.max(output, 1)
        predict = predict + 1
        target = target + 1

        predict = predict * (target > 0).long()
        intersection = predict * (predict == target).long()

        area_inter = torch.histc(intersection.float(), bins=self.num_classes, max=self.num_classes, min=1)
        area_pred = torch.histc(predict.float(), bins=self.num_classes, max=self.num_classes, min=1)
        area_lab = torch.histc(target.float(), bins=self.num_classes, max=self.num_classes, min=1)
        area_union = area_pred + area_lab - area_inter
        iou = 1.0 * area_inter / (np.spacing(1) + area_union)
        return (2 * iou) / (iou + 1)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):

        ce = F.cross_entropy(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.contiguous().view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * ce + dice