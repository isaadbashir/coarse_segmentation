import config
import training.models.resnet_dilated as dilated_resnet
import training.models.resnet_normal as normal_resnet
import training.models.mobilenet_dilated as dilated_mbnet
import torch.nn as nn
import torch

class ModelBuilder(nn.Module):

    def __init__(self, model_name, mini_patch_size, num_classes) -> None:
        super(ModelBuilder, self).__init__()

        self.model_name = model_name
        self.mini_patch_size = mini_patch_size
        self.num_classes = num_classes

        self.encoder, self.out_channels = self.get_model()
        self.classifier = nn.Sequential(nn.Dropout(0.1),
                                        nn.Conv2d(self.out_channels, self.num_classes, kernel_size=1, stride=1))

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
                                        output_stride=self.mini_patch_size), 2048

    def get_normal_resnet(self):
        return normal_resnet.ResNet50(BatchNorm=nn.BatchNorm2d,
                                        pretrained=True,
                                        output_stride=self.mini_patch_size), 2048

    def get_normal_mobilenet(self):
        return dilated_mbnet.MobileNetV2(BatchNorm=nn.BatchNorm2d,
                                            pretrained=True,
                                            output_stride=self.mini_patch_size), 320

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.encoder = nn.DataParallel(self.encoder, device_ids=device_ids).to('cuda:0')
        else:
            self.encoder = self.encoder.to(device)

        self.classifier = self.classifier.to(device)

    def forward(self, x):
        enc_out,_ = self.encoder(x)
        output = self.classifier(enc_out)

        return output

    
