from functools import partial
import os
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pickle
import config
import utils

def create_coarse_patches(path, pre_processing_fun = None):

    # load the patch
    patch = np.load(path, allow_pickle=True)

    # seperate image and mask
    image, mask = patch[:, :, :3], patch[:, :, 3]

    # apply preprocessing function on the masks
    if pre_processing_fun is not None:
        mask = pre_processing_fun(mask)

    # ori_mask = np.copy(mask)
    output_size = mask.shape[0]//config.COARSE_FILTER_SIZE

    coarse_mask = np.zeros((output_size,output_size), dtype=np.uint8)

    for i in range(0, mask.shape[0], config.COARSE_FILTER_SIZE):

        for j in range(0, mask.shape[1], config.COARSE_FILTER_SIZE):

            try:

                # to deal with the corner cases where the size is not complete patch
                out_margin_y = 0
                out_margin_x = 0
                y_slice = i + config.COARSE_FILTER_SIZE
                x_slice = j + config.COARSE_FILTER_SIZE

                # find out the corner margins available
                if mask.shape[0] + 1 <= y_slice:
                    out_margin_y = y_slice - mask.shape[0]
                    y_slice = y_slice - out_margin_y
                if mask.shape[1] + 1 <= x_slice:
                    out_margin_x = x_slice - mask.shape[1]
                    x_slice = x_slice - out_margin_x

                # apply the max in the window size
                mask[i:y_slice - out_margin_y, j:x_slice - out_margin_x] = np.max(mask[i:y_slice - out_margin_y, j:x_slice - out_margin_x])
                coarse_mask[i//config.COARSE_FILTER_SIZE, j//config.COARSE_FILTER_SIZE] = np.max(mask[i:y_slice - out_margin_y, j:x_slice - out_margin_x])

            except Exception as ex:
                print(f'{ex}\n Name: {path}\n I:J {i}{j}')

    # due to the variable shape of the coarse mask w need to store it as dict
    small_combo = {config.KEY_IMAGE: image,
                    config.KEY_COARSE_MASK_RESIZE: mask,
                    config.KEY_COARSE_MASK: coarse_mask,
                    config.KEY_MASK: patch[:, :, 3] if pre_processing_fun is None else pre_processing_fun(patch[:, :, 3])}

    # exrtact the meta data for saving files
    patch_name = path.split('/')[-1]
    patch_split = path.split('/')[-2]

    # output path
    OUTPUT_PATH = os.path.join(config.COARSEWISE_PATCHES_PATH,
                                config.PATCHES_MAGNIFICATION_LEVEL,
                                f'{config.COARSE_FILTER_SIZE}x{config.COARSE_FILTER_SIZE}',
                                patch_split)

    # plt.subplot(1,3,1)
    # plt.imshow(image)
    
    # plt.subplot(1,3,2)
    # plt.imshow(coarse_mask)
    
    # plt.subplot(1,3,3)
    # plt.imshow(mask)
    
    # plt.show()

    # create the directories needed
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH, exist_ok=True)

    with open(f'{OUTPUT_PATH}/{patch_name[:-4]}.pkl', 'wb') as f:
        pickle.dump(small_combo, f, pickle.HIGHEST_PROTOCOL)


def main():

    # get list of all folders i.e., train, valid, test etc
    list_of_folders = os.listdir(os.path.join(config.PIXELWISE_PATCHES_PATH,config.PATCHES_MAGNIFICATION_LEVEL))

    for split in list_of_folders:

        print(f'Starting the {split} folder .... ')
        INPUT_PATH = os.path.join(config.PIXELWISE_PATCHES_PATH, config.PATCHES_MAGNIFICATION_LEVEL, split)
        list_of_patches = os.listdir(INPUT_PATH)

        mp = Pool(20)
        mp.map(partial(create_coarse_patches, pre_processing_fun = utils.group_mask_tnbc_classes),
                        [os.path.join(INPUT_PATH,i) for i in list_of_patches])


if __name__ == '__main__':
    main()


    
