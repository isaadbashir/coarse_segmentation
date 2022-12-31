from functools import partial
import os
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pickle
import src.config as config
import src.utils as utils
import cv2
import staintools


PATH_TO_REF_IMAGE = '/mnt/sda2/tnbc_segmetnation_v2/dataset/tnbc/roi/roi_20x/train/images/TCGA-A7-A26G-DX1_xmin77919_ymin10728_MAG-20.00.png'


def create_coarse_patches(path, pre_processing_fun = None):

    # load the patch
    patch = np.load(path, allow_pickle=True)

    # INFO: due to previously wrong format saved for 20x patches need to do channle switch 
    # seperate image and mask
    if config.PATCHES_MAGNIFICATION_LEVEL != '20x':
        image, mask = patch[:, :, :3], patch[:, :, 3]
    else:
        image, mask = cv2.cvtColor(patch[:, :, :3], cv2.COLOR_BGR2RGB), patch[:, :, 3]

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

    # normalize the image
    target = staintools.read_image(PATH_TO_REF_IMAGE)
    normalizer = staintools.StainNormalizer('macenko')
    normalizer.fit(target)
    normalized_image = normalizer.transform(image)

    # augment the iamge
    list_of_augmented_images = []
    list_of_augmented_images.append(image)
    list_of_augmented_images.append(normalized_image)
    
    augmentor1 = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2)
    augmentor1.fit(image)
    for i in range(3):
        augmented_image = augmentor1.pop().astype(np.uint8)
        list_of_augmented_images.append(augmented_image)

    augmentor1.fit(normalized_image)
    for i in range(3):
        augmented_image = augmentor1.pop().astype(np.uint8)
        list_of_augmented_images.append(augmented_image)

    augmentor2 = staintools.StainAugmentor(method='macenko', sigma1=0.2, sigma2=0.2)
    augmentor2.fit(image)
    for i in range(3):
        augmented_image = augmentor2.pop().astype(np.uint8)
        list_of_augmented_images.append(augmented_image)

    augmentor2.fit(normalized_image)
    for i in range(3):
        augmented_image = augmentor2.pop().astype(np.uint8)
        list_of_augmented_images.append(augmented_image)

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

    original_mask = patch[:, :, 3] if pre_processing_fun is None else pre_processing_fun(patch[:, :, 3])

    np.savez(f'{OUTPUT_PATH}/{patch_name[:-4]}.npz', *list_of_augmented_images, mask = mask, original_mask = original_mask)




def main():

    # get list of all folders i.e., train, valid, test etc
    list_of_folders = os.listdir(os.path.join(config.PIXELWISE_PATCHES_PATH,config.PATCHES_MAGNIFICATION_LEVEL,f'1x1'))

    for split in list_of_folders:

        print(f'Starting the {split} folder .... ')
        INPUT_PATH = os.path.join(config.PIXELWISE_PATCHES_PATH, config.PATCHES_MAGNIFICATION_LEVEL, f'1x1',split)
        list_of_patches = os.listdir(INPUT_PATH)

        mp = Pool(20)
        mp.map(partial(create_coarse_patches, pre_processing_fun = utils.group_mask_tnbc_classes),
                        [os.path.join(INPUT_PATH,i) for i in list_of_patches])
        # create_coarse_patches(os.path.join(INPUT_PATH, list_of_patches[0]), utils.group_mask_tnbc_classes)


if __name__ == '__main__':
    main()


    
