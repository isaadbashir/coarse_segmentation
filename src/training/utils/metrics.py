import numpy as np
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)

def batch_pix_accuracy(output, target):
    _, predict = torch.max(output, 1)

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_confusion_matrix(output, target, num_class):
    _, predict = torch.max(output, 1)

    predict = predict.int()
    target = target.int()

    predict = predict.flatten()
    target = target.flatten()

    y = num_class * target + predict
    y = torch.bincount(y)
    if len(y) < num_class * num_class:
        y = torch.cat((y, torch.zeros(num_class * num_class - len(y), dtype=torch.long).cuda()))
    y = y.reshape(num_class, num_class).cpu().numpy()
    
    return y


def batch_intersection_union(output, target, num_class):
    _, predict = torch.max(output, 1)
    predict = predict + 1
    target = target + 1

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    
    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def eval_metrics(output, target, num_classes, ignore_index = 255):
    target = target.clone()
    target[target == ignore_index] = -1
    correct, labeled = batch_pix_accuracy(output.data, target)
    inter, union = batch_intersection_union(output.data, target, num_classes)
    confusion_matrix = batch_confusion_matrix(output.data, target, num_classes)
    return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5)], confusion_matrix