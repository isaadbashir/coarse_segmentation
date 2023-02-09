import os
import pickle
import random

import albumentations as album
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset as BaseDataset
import src.config as config
import src.utils as utils
import cv2
import time

class DataReader(BaseDataset):

    def __init__(self, data_dir, target_size, num_classes, transformation=None):

        self.data_dir = data_dir
        self.target_size = target_size
        self.num_classes = num_classes

        if transformation is None:
            self.transform = album.Compose([
                album.Resize(height=target_size, width=target_size),
                album.Normalize(
                        mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0),
                ToTensorV2(),
            ], additional_targets={'image0': 'image'})
            
        else:
            self.transform = transformation

        # create the lists of images with full path
        self.list_of_images = [os.path.join(self.data_dir, image) for image in os.listdir(self.data_dir)]

        # shuffle the list of images for randomness
        random.shuffle(self.list_of_images)

    def preprocess_for_train(self, image, aug_image, mask):
        '''
        preprocess training images with albumenations
        :param image:
        :param mask:
        :return:
        '''

        random.seed(11)
        augmented = self.transform(image=image, mask = mask, image0=aug_image)

        return augmented['image'], augmented['image0'], augmented['mask']

    def __getitem__(self, i):

        random.seed(time.time())
        # read data which is in ND format where first 3 are image and 4th is mask
        image_data = np.load(self.list_of_images[i])
        
        image = image_data['arr_0'] # index of the stained image
        mask = image_data['pixel_mask'] # index of the mask
        aug_image = image_data['arr_1'] # index of the augmetned image
        # apply transformations
        image, aug_image, mask = self.preprocess_for_train(image, aug_image, mask)

        # convert mask from HW to CHW mask
        #mask = torch.nn.functional.one_hot(mask.to(torch.int64), num_classes=self.num_classes)

        return image, aug_image, mask.long(), self.list_of_images[i].split('/')[-1].split('_')[0]

    def __len__(self):
        return len(self.list_of_images)


if __name__ == '__main__':

    data_dir = '/mnt/sda2/coarse_segmentation/data/patches/augmented_coarse_wise/20x/16x16/train/'
    target_size = 512
   
    transform = album.Compose([
        album.Resize(height=target_size, width=target_size),
        album.VerticalFlip(p=0.8),
        album.HorizontalFlip(p=0.8),
        album.Rotate(limit=75, p=1.0),
        album.RandomBrightnessContrast(p=0.8),
        album.RandomGamma(p=0.8),
        album.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0),
        ToTensorV2()
    ],additional_targets={'image0': 'image'})

    params = {'batch_size': 12,
              'shuffle': False,
              'num_workers': 0,
              'pin_memory': False}

    data_reader = DataReader(data_dir=data_dir,
                            num_classes=config.TNBC_NUMBER_OF_CLASSES,
                            target_size=target_size,
                            transformation=transform)

    train_generator = data.DataLoader(data_reader, **params)

    X, AX, Y, N = next(iter(train_generator))
    print(X.shape, Y.shape)

    # fig = utils.plot_image_prediction(X.cpu().numpy(), Y.numpy(), F.one_hot(Y.to(torch.int64), num_classes=config.TNBC_NUMBER_OF_CLASSES).permute(0,3,1,2).numpy(), N, 5, 5)
    fig = utils.plot_image_prediction(AX.cpu().numpy(), Y.numpy(), X.cpu().numpy(), N, 12, 5)

    plt.show()
