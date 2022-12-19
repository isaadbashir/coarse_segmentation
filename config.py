# global path  variables
PIXELWISE_PATCHES_PATH = '/mnt/sda2/coarse_segmentation/data/patches/pixel_wise/'
COARSEWISE_PATCHES_PATH = '/mnt/sda2/coarse_segmentation/data/patches/coarse_wise/'
PATCHES_MAGNIFICATION_LEVEL = '20x' # must of be string as 10x, 20x, 40x

## pre-processing
# coarse masks
# filter size i.e., the window max will be taken and assigned
COARSE_FILTER_SIZE = 32


# common functions used in the project
def group_mask_tnbc_classes(mask):
    """
    Groups the minor mask classes into major ones according to the info given in the paper
    :param mask:
    """
    # tumor cluster
    mask[mask == 19] = 1
    mask[mask == 20] = 1

    # inflammatory
    mask[mask == 10] = 3
    mask[mask == 11] = 3

    # others
    mask[mask == 5] = 5
    mask[mask == 6] = 5
    mask[mask == 7] = 5
    mask[mask == 8] = 5
    mask[mask == 9] = 5
    mask[mask == 10] = 5
    mask[mask == 11] = 5
    mask[mask == 12] = 5
    mask[mask == 13] = 5
    mask[mask == 14] = 5
    mask[mask == 15] = 5
    mask[mask == 16] = 5
    mask[mask == 17] = 5
    mask[mask == 18] = 5
    mask[mask == 19] = 5
    mask[mask == 20] = 5
    mask[mask == 21] = 5

    mask[mask == 5] = 0

    return mask