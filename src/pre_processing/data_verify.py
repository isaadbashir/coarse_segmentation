import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

PATH_TO_ROIS = '/mnt/sdb2/TNBC/dataset/0_Public-data-Amgad2019_0.50MPP/images/'
PATH_TO_MASKS = '/mnt/sdb2/TNBC/dataset/0_Public-data-Amgad2019_0.50MPP/masks/'

# loop though images and masks and plot them using matplotlib

list_of_images = os.listdir(PATH_TO_ROIS)
list_of_masks = os.listdir(PATH_TO_MASKS)

for i in range(len(list_of_images)):
    image = cv2.imread(os.path.join(PATH_TO_ROIS, list_of_images[i]))[..., ::-1] # change BGR to RGB
    mask = cv2.imread(os.path.join(PATH_TO_MASKS, list_of_masks[i]),0) # read the grayscale mask

    # to keep the masks with same labels for visualization 
    for c in range(20):
        mask[0,c] = c

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()