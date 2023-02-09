import config
import training.models.resnet_dilated as dilated_resnet
import training.models.resnet_normal as normal_resnet
import training.models.mobilenet_dilated as dilated_mbnet
import training.models.efficient_net as normal_efficient
import torch.nn as nn
import torch
import training.models.unext as unext
from training.models.deeplab import DeepLab

class ModelBuilder(nn.Module):

    def __init__(self, model_name, mini_patch_size, input_size,  num_classes, is_ema = False) -> None:
        super(ModelBuilder, self).__init__()

        self.model_name = model_name
        self.mini_patch_size = mini_patch_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.encoder, self.out_channels = self.get_model()
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                                        nn.Conv2d(self.out_channels, self.out_channels//2, kernel_size=1, stride=1),
                                        nn.BatchNorm2d(self.out_channels//2),
                                        nn.ReLU(),
                                        nn.Conv2d(self.out_channels//2, self.num_classes, kernel_size=1, stride=1))

        if is_ema:
            for param in self.encoder.parameters():
                param.detach_()
            for param in self.classifier.parameters():
                param.detach_()
    
    def update_ema_variables(self, model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(self.encoder.parameters(), model.encoder.parameters()):
            ema_param.data.mul_(alpha).add_( param.data, alpha= 1 - alpha)
        for ema_param, param in zip(self.classifier.parameters(), model.classifier.parameters()):
            ema_param.data.mul_(alpha).add_( param.data, alpha= 1 - alpha)


    def get_model(self):

        if self.model_name == config.NORMAL_RESNET_MODEL:
            return self.get_normal_resnet()
        elif self.model_name == config.DILATED_RESNET_MODEL:
            return self.get_dilated_resnet()
        elif self.model_name == config.DILATED_MOBILENET_MODEL:
            return self.get_normal_mobilenet()
        elif self.model_name == config.NORMAL_EFFICIENT_NET_B5:
            return self.get_normal_eficient_b5()
        elif self.model_name == config.NORMAL_EFFICIENT_NET_B4:
            return self.get_normal_eficient_b4()
        elif self.model_name == config.NORMAL_EFFICIENT_NET_B3:
            return self.get_normal_eficient_b3()
        elif self.model_name == config.NORMAL_EFFICIENT_NET_B2:
            return self.get_normal_eficient_b2()
        elif self.model_name == config.NORMAL_EFFICIENT_NET_B1:
            return self.get_normal_eficient_b1()
        elif self.model_name == config.NORMAL_EFFICIENT_NET_B0:
            return self.get_normal_eficient_b0()
        elif self.model_name == config.UNET_MODEL:
            return unext.UNext(), 100
        elif self.model_name == config.DEEPLAB_MODEL:
            return DeepLab(backbone='resnet50', output_stride=16), 256
        else:
            raise Exception(f"{self.model_name} dosen't exist ")
    
    def get_dilated_resnet(self):
        return dilated_resnet.ResNet50(BatchNorm=nn.BatchNorm2d,
                                        pretrained=True,
                                        output_stride=self.mini_patch_size), 2048#3904

    def get_normal_resnet(self):
        return normal_resnet.ResNet50(BatchNorm=nn.BatchNorm2d,
                                        pretrained=True,
                                        output_stride=self.mini_patch_size), 2048

    def get_normal_mobilenet(self):
        return dilated_mbnet.MobileNetV2(BatchNorm=nn.BatchNorm2d,
                                            pretrained=True,
                                            output_stride=self.mini_patch_size), 320

    def get_normal_eficient_b5(self):
        return normal_efficient.EfficientNet.from_pretrained(f'efficientnet-b5',
                                                            output_stride = self.mini_patch_size,
                                                            num_classes=5, 
                                                            image_size = self.input_size,
                                                            include_top = False), 2048

    def get_normal_eficient_b4(self):
        return normal_efficient.EfficientNet.from_pretrained(f'efficientnet-b4',
                                                            output_stride = self.mini_patch_size,
                                                            num_classes=5, 
                                                            image_size = self.input_size,
                                                            include_top = False), 1792

    def get_normal_eficient_b3(self):
        return normal_efficient.EfficientNet.from_pretrained(f'efficientnet-b3',
                                                            output_stride = self.mini_patch_size,
                                                            num_classes=5, 
                                                            image_size = self.input_size,
                                                            include_top = False), 1536
    def get_normal_eficient_b2(self):
        return normal_efficient.EfficientNet.from_pretrained(f'efficientnet-b2',
                                                            output_stride = self.mini_patch_size,
                                                            num_classes=5, 
                                                            image_size = self.input_size,
                                                            include_top = False), 1408
 
    def get_normal_eficient_b1(self):
        return normal_efficient.EfficientNet.from_pretrained(f'efficientnet-b1',
                                                            output_stride = self.mini_patch_size,
                                                            num_classes=5, 
                                                            image_size = self.input_size,
                                                            include_top = False), 1280

    def get_normal_eficient_b0(self):
        return normal_efficient.EfficientNet.from_pretrained(f'efficientnet-b0',
                                                            output_stride = self.mini_patch_size,
                                                            num_classes=5, 
                                                            image_size = self.input_size,
                                                            include_top = False), 1280

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.encoder = nn.DataParallel(self.encoder, device_ids=device_ids).to('cuda:0')
        else:
            self.encoder = self.encoder.to(device)

        self.classifier = self.classifier.to(device)

    def forward(self, x):

        if self.model_name == config.UNET_MODEL:
            return self.encoder(x)
        
        if self.model_name == config.DEEPLAB_MODEL:
            enc_out = self.encoder(x)
            return self.classifier(enc_out), enc_out


        enc_out,_ = self.encoder(x)
        output = self.classifier(enc_out)

        return output, enc_out

    
