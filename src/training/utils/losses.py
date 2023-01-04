import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def CE_loss(input_logits, target_targets, ignore_index = 255, temperature=1):
    return F.cross_entropy(input_logits / temperature, target_targets, ignore_index=ignore_index)


def mse_loss(inputs, targets):
    return F.mse_loss(inputs, targets, reduction='mean') 

def kl_loss(inputs, targets):
    return F.kl_div(inputs, targets, reduction='mean')


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


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss