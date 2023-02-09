import torch
import torch.nn as nn
import torch.nn.functional as F
# import training.models.aspp as aspp
# import training.models.decoder as decoder
# import training.models.resnet_normal as backbone
from .aspp import build_aspp
from .decoder import build_decoder
from .resnet_dilated import ResNet50

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet101', output_stride=16, num_classes=21,
                 sync_bn=False, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        assert sync_bn == False
        assert freeze_bn == False

        BatchNorm = nn.BatchNorm2d
        self.freeze_bn = freeze_bn

        self.backbone = ResNet50(BatchNorm=BatchNorm, pretrained=True, output_stride=output_stride)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)


    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_backbone_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_module_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='resnet50', output_stride=16)
    print(model)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


