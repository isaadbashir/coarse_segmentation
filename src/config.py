import logging

# global path  variables
PIXELWISE_PATCHES_PATH = '/mnt/sda2/coarse_segmentation/data/patches/pixel_wise/'
COARSEWISE_PATCHES_PATH = '/mnt/sda2/coarse_segmentation/data/patches/coarse_wise/'
PATCHES_MAGNIFICATION_LEVEL = '20x' # must of be string as 10x, 20x, 40x

# number of classes
TNBC_NUMBER_OF_CLASSES = 5

## pre-processing
# coarse masks
# filter size i.e., the window max will be taken and assigned [8,16,32]
COARSE_FILTER_SIZE = 8

# dict keys for saving patches
# image
KEY_IMAGE = 'img'
# original mask
KEY_MASK = 'omask'
# coarse mask smaller
KEY_COARSE_MASK = 'cmask'
# coarse mask of size as orignial mask
KEY_COARSE_MASK_RESIZE = 'mask'

# model names for usage
NORMAL_RESNET_MODEL = 'resnet'
DILATED_RESNET_MODEL = 'dilated_resnet'
DILATED_MOBILENET_MODEL = 'dilated_mb_net'
UNET_MODEL = 'unet'
DEEPLAB_MODEL = 'deeplab'
SEGFORMER_MODEL = 'segformer'

# logger
LOG_FORMATTER = logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s")