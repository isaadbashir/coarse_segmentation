import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import src.config as config

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(config.LOG_FORMATTER)
    return console_handler


def get_file_handler(file_path):
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(config.LOG_FORMATTER)
    return file_handler


def get_logger(logger_name, file_path):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler(file_path))
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger


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



def create_mask(pred_mask):
    pred_mask = np.argmax(pred_mask, axis=-1)
    return pred_mask


def plot_image_prediction(images, masks, outputs, names, num_of_images, num_of_classes, font_size=14):

    # convert to numpy array
    images = np.array(images)
    masks = np.array(masks)
    outputs = np.array(outputs)

    # permute the images
    images = images.transpose(0, 2, 3, 1)
    outputs = outputs.transpose(0, 2, 3, 1)

    pred_mask = create_mask(outputs)
    # masks = create_mask(masks)

    fig, ax = plt.subplots(num_of_images, 4, figsize=(8, 24))

    ax[0, 0].set_title("Image", fontsize=font_size)
    ax[0, 1].set_title("Mask ", fontsize=font_size)
    ax[0, 2].set_title("Pred Mask", fontsize=font_size)
    ax[0, 3].set_title("Overlay", fontsize=font_size)

    for i in range(num_of_images):

        ax[i, 0].imshow(images[i, ...])

        mask_i = np.copy(masks[i, ...])
        pred_mask_i = np.copy(pred_mask[i, ...])

        mask_i[0, 0] = 0
        mask_i[0, 1] = 1
        mask_i[0, 2] = 2
        mask_i[0, 4] = 3
        mask_i[0, 5] = 4

        ax[i, 1].imshow(mask_i.astype('uint8'))

        pred_mask_i[0, 0] = 0
        pred_mask_i[0, 1] = 1
        pred_mask_i[0, 2] = 2
        pred_mask_i[0, 4] = 3
        pred_mask_i[0, 5] = 4

        ax[i, 2].imshow(pred_mask_i.astype('uint8'))

        pred_mask_i_overlay = np.ma.masked_where(pred_mask_i == 0, pred_mask_i)

        ax[i, 3].imshow(images[i, ...])
        ax[i, 3].imshow(pred_mask_i_overlay.astype('uint8'), interpolation='none', cmap='jet', alpha=0.5)

        ax[i, 0].set_ylabel(names[i], fontsize=6)
        ax[i, 0].set_yticks([])
        ax[i, 0].set_xticks([])

        ax[i, 1].set_yticks([])
        ax[i, 1].set_xticks([])

        ax[i, 2].set_yticks([])
        ax[i, 2].set_xticks([])

        ax[i, 3].set_yticks([])
        ax[i, 3].set_xticks([])

    return fig

