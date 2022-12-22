import config
import models.resnet_dilated as dilated_resnet
import models.resnet_normal as normal_resnet
import models.mobilenet_dilated as dilated_mbnet
import torch.nn as nn

class ModelBuilder:

    def __init__(self, model_name, mini_patch_size) -> None:
        self.model_name = model_name
        self.mini_patch_size = mini_patch_size

    def get_model(self):

        if self.model_name == config.NORMAL_RESNET_MODEL:
            return self.get_normal_resnet()
        elif self.model_name == config.DILATED_RESNET_MODEL:
            return self.get_dilated_resnet()
        elif self.model_name == config.DILATED_MOBILENET_MODEL:
            return self.get_normal_mobilenet
        elif self.model_name == config.UNET_MODEL:
            pass
        elif self.model_name == config.DEEPLAB_MODEL:
            pass
        else:
            raise Exception(f"{self.model_name} dosen't exist ")
    
    def get_dilated_resnet(self):
        return dilated_resnet.ResNet50(BatchNorm=nn.BatchNorm2d,
                                        pretrained=True,
                                        output_stride=self.mini_patch_size)

    def get_normal_resnet(self):
        return normal_resnet.ResNet50(BatchNorm=nn.BatchNorm2d,
                                        pretrained=True,
                                        output_stride=self.mini_patch_size)

    def get_normal_mobilenet(self):
        return dilated_mbnet.MobileNetV2(BatchNorm=nn.BatchNorm2d,
                                            pretrained=True,
                                            output_stride=self.mini_patch_size)

    
