import config


class ModelBuilder:

    def __init__(self, model_name, mini_patch_size) -> None:
        self.model_name = model_name
        self.mini_patch_size = mini_patch_size

    def get_model(self):

        if self.model_name == config.RESNET_MODEL:
            pass
        elif self.model_name == config.MOBILENET_MODEL:
            pass
        elif self.model_name == config.UNET_MODEL:
            pass
        elif self.model_name == config.DEEPLAB_MODEL:
            pass
        else:
            raise Exception(f"{self.model_name} dosen't exist ")
    