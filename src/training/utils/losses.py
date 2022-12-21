import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def CE_loss(input_logits, target_targets, ignore_index, temperature=1):
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