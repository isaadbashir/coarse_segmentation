import copy
import os
import random
import albumentations as album
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchsummary
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from training.data.data_reader import DataReader
import config
from training.utils.losses import *
from training.utils.metrics import *
from training.models.model_builder import ModelBuilder
from utils import *
from time import time

class Trainer:
    """
    Training the coarse segmentation networks

    """

    def __init__(self, model_name, data_path, mini_patch_size, patch_size, patch_magnification, output_path, num_classes, class_names, num_epochs, batch_size, gpu_id, experiment_no):
        self.model_name = model_name
        self.data_path = data_path
        self.patch_size = patch_size
        self.mini_patch_size = mini_patch_size
        self.patch_magnification = patch_magnification
        self.output_path = output_path
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.experiment_no = f"{experiment_no}_{str(time()).split('.')[0]}"
        self.config_no = self.experiment_no.split('_')[-1]
        self.class_names = class_names

        # create directories path
        self.log_dir = os.path.join(self.output_path, 'logs', self.experiment_no)
        self.tensorboard_dir = os.path.join(self.output_path, 'tensorboard', self.experiment_no)
        self.models_dir = os.path.join(self.output_path, 'models', self.experiment_no)
        self.raw_dir = os.path.join(self.output_path, 'raw', self.experiment_no)

        # create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)

        # setup logging and device to be used
        self.logger = get_logger(__name__, os.path.join(self.log_dir, f'{self.__class__.__name__}.log'))
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")

        # setup the tensorboard here
        self.writer = SummaryWriter(os.path.join(self.tensorboard_dir, f'{self.__class__.__name__}'))

        # set the gpu for usage
        torch.cuda.set_device(self.gpu_id)

    def load_model(self):
        '''
        load the coarse semgentation models for training the baseline
        :return: model
        '''

        model = ModelBuilder(self.model_name, self.mini_patch_size, self.num_classes)
     
        # put model to the GPU
        model.relocate()

        return model

    def load_data(self):
        """
        load the data and returns the train and test generators

        :return: train and test generators
        """

        # create the train augmentation
        self.logger.info(f'STEP:2 - Creating augmentation for train and test')
        train_transform = album.Compose([
            album.Resize(height=self.patch_size, width=self.patch_size),
            album.VerticalFlip(p=0.5),
            album.HorizontalFlip(p=0.5),
            album.Rotate(limit=75, p=0.5),
            album.RandomBrightnessContrast(p=0.5),
            album.RandomGamma(p=0.5),
            album.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0),
            ToTensorV2()
        ])

        # create the test augmentation
        test_transform = album.Compose([
            album.Resize(height=self.patch_size, width=self.patch_size),
            album.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0),
            ToTensorV2()
        ])

        # create the data loader params
        params = {'batch_size': self.batch_size,
                  'shuffle': False,
                  'num_workers': 4,
                  'pin_memory': False}
        self.logger.info(f'STEP:2 - Creating params for data loaders {params}')
        
        # create the paths for the train and test
        train_patch_path = os.path.join(self.data_path, f'{self.patch_magnification}', f'{self.mini_patch_size}x{self.mini_patch_size}', 'train')
        test_patch_path = os.path.join(self.data_path, f'{self.patch_magnification}', f'{self.mini_patch_size}x{self.mini_patch_size}', 'test')

        # create dataset reader
        data_reader = DataReader(data_dir=train_patch_path,
                                    target_size=self.patch_size,
                                    num_classes=self.num_classes,
                                    transformation=train_transform)

        data_test_reader = DataReader(data_dir=test_patch_path,
                                        target_size=self.patch_size,
                                        num_classes=self.num_classes,
                                        transformation=test_transform)

        # create the data generator for pytorch
        self.logger.info(f'STEP:2 - Loading the data files from \ntrain:{train_patch_path} \ntest: {test_patch_path}')
        train_generator = data.DataLoader(data_reader, **params)
        test_generator = data.DataLoader(data_test_reader, **params)

        return train_generator, test_generator

    def get_optim(self, model, opt, lr, reg=0.0001, params=False):
        """
        returns the optimizer based on the params provided and handles the pytorch's requirements
        :param model:
        :param opt:
        :param lr:
        :param reg:
        :param params:
        :return:
        """
        if params:
            temp = model
        else:
            temp = filter(lambda p: p.requires_grad, model.parameters())

        if opt == "adam":
            optimizer = torch.optim.Adam(temp, lr=lr, weight_decay=reg)
        elif opt == 'sgd':
            optimizer = torch.optim.SGD(temp, lr=lr, momentum=0.9, weight_decay=reg)
        else:
            raise NotImplementedError

        return optimizer

    def initialize(self):

        self.loss = AverageMeter()
        self.total_inter = 0 
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        self.mIoU = []
        self.mDice = []
        self.pixel_acc = 0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.class_iou = {}
        self.class_dice = {}

    def update_loss(self, loss):
        n = loss.numel()
        count = torch.tensor([n]).long().cuda()
        n = count.item()
        mean = loss.sum() / n
        self.loss.update(mean.item())

    def update_metrics(self, correct, labeled, inter, union, confusion_matrix):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
        self.confusion_matrix += confusion_matrix

    def get_metrics(self):
     
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        dice = (2 * IoU) / (IoU + 1)
        
        mIoU = IoU.mean()
        mDice = dice.mean()

        confusion_matrix = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis]

        return {
            "Pixel_Accuracy": np.round(pixAcc, 2),
            "Mean_IoU": np.round(mIoU, 2),
            "Mean_Dice": np.round(mDice, 2),
            "Class_IoU": dict(zip(self.class_names, np.round(IoU, 2))),
            "Class_Dice": dict(zip(self.class_names, np.round(dice, 2))),
            "Confusion_Matrix": confusion_matrix
        }

    def train_model(self):
        """
        train loop for training the model which includes creating the model,
        reading data and training the model using the data

        """

        # load the model
        self.logger.info(f'STEP:1 - Loading model {self.model_name}')
        model = self.load_model()
        self.logger.info(f'STEP:1 - Model {self.model_name} loaded successfully')

        # load the data generators
        self.logger.info(f'STEP:2 - Loading data generators ')
        train_generator, test_generator = self.load_data()
        self.logger.info(f'STEP:2 - Data generators loaded successfully')

        self.logger.info(f'STEP:3 - Loading optimizers')
        optim = self.get_optim(model, 'adam', 0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=0.00001)
        self.logger.info(f'STEP:3 - Loaded \n{optim} successfully')

        self.logger.info(f'STEP:4 - Loading losses')
        loss = CE_loss
        self.logger.info(f'STEP:4 - Loading losses successfully')

        # progress bar outer for epoch
        self.logger.info(f'STEP:5 - Starting the training loop')
        pbar_out_train = tqdm(total=self.num_epochs, position=0)
        pbar_out_test = tqdm(total=self.num_epochs, position=2)

        # progress bar inner for epoch batches
        pbar_train = tqdm(total=len(train_generator), position=1)
        pbar_test = tqdm(total=len(test_generator), position=3)

        best_f1 = 0
        best_model = None
        best_epoch = 0

        # loop through the epochs
        for epoch in range(self.num_epochs):

            phase = 'train'

            # run the train step
            epoch_loss, epoch_accuracy, epoch_iou, epoch_f1, epoch_class_iou, epoch_class_dice = self.model_step(epoch,
                                                                                phase, 
                                                                                model, 
                                                                                train_generator, 
                                                                                optim, 
                                                                                loss, 
                                                                                scheduler, 
                                                                                pbar_train)

            # save the best model
            if best_f1 < epoch_f1:
                best_f1 = epoch_f1
                best_model = copy.deepcopy(model)
                best_epoch = epoch

            train_string = f'Main Loop  - {phase}: Epoch:{epoch}' \
                            f' | Loss:{epoch_loss:.4f}' \
                            f' | IOU:{epoch_iou:.4f}' \
                            f' | F1:{epoch_f1:.4f}' \
                            f' | Accuracy:{epoch_accuracy:.4f}' \
                            f' | IOU:{epoch_class_iou}' \
                            f' | Dice:{epoch_class_dice}'


            pbar_out_train.set_description(train_string)

            phase = 'test'

            # run the test step
            epoch_loss, epoch_accuracy, epoch_iou, epoch_f1, epoch_class_iou, epoch_class_dice = self.model_step(epoch, 
                                                                                phase,
                                                                                model, 
                                                                                test_generator, 
                                                                                optim, 
                                                                                loss, 
                                                                                scheduler, 
                                                                                pbar_test)

            test_string = f'Main Loop  - {phase}: Epoch:{epoch}' \
                            f' | Loss:{epoch_loss:.4f}' \
                            f' | IOU:{epoch_iou:.4f}' \
                            f' | F1:{epoch_f1:.4f}' \
                            f' | Accuracy:{epoch_accuracy:.4f}' \
                            f' | IOU:{epoch_class_iou}' \
                            f' | Dice:{epoch_class_dice}'

            pbar_out_test.set_description(test_string)

            pbar_out_train.update(1)
            pbar_out_test.update(1)


            print('\n\n\n\n')
            print('-' * 180)
            self.logger.info(f'Epoch {epoch}/{self.num_epochs - 1}')
            self.logger.info(train_string)
            self.logger.info(test_string)
            print('-' * 180)
            print('\n')

        self.logger.info(f'Running the saving the best model with f1-score {best_f1} at epoch {best_epoch}')

        # save the best model
        save_path = os.path.join(self.models_dir, f'epoch_{best_epoch}_f1_{best_f1}_model.pth')
        os.makedirs(save_path, exist_ok=True)
        torch.save(best_model, f'{save_path}/model.pt')

        # run the last epoch as best model for visualization
        phase = 'train'
        # run the train step
        self.model_step(self.num_epochs, 
                        phase, 
                        best_model, 
                        train_generator, 
                        optim, 
                        loss, 
                        scheduler, 
                        pbar_train)

        phase = 'test'
        # run the test step
        self.model_step(self.num_epochs, 
                        phase, 
                        best_model, 
                        test_generator, 
                        optim, 
                        loss, 
                        scheduler, 
                        pbar_test)

    def model_step(self, epoch, phase, model, generator, optimizer, criterion, scheduler, pbar):
        """
        Runs the one epoch with train or test phase and avoid the duplication of the same code
        :param epoch:
        :param phase:
        :param model:
        :param generator:
        :param optimizer:
        :param loss:
        :return:
        """

        # reset the and initialize the metrics again
        self.initialize()

        # check the training phase 
        if phase == 'train':
            # set the model in training mode
            model.train()
        else:
            model.eval()

        # TODO: used to get top,worst and random images
        all_f1_scores = []

        # load the batch
        for images, masks, _ in generator:

            # move the batch to cuda device as the model is on cuda
            images = images.to(self.device)
            masks = masks.to(self.device)

            # permute the masks to match the output
            # masks = masks.permute(0, 3, 1, 2)

            # reset the optimizer gradients to zero for next epoch calculations
            optimizer.zero_grad()

            # for safe check make sure gradients are enabled by working in enabled scope
            with torch.set_grad_enabled(phase == 'train'):

                # pass the images through the model and get output
                output = model(images)

                # resize the output to 512 for consistency
                output = F.interpolate(output, size = (self.patch_size, self.patch_size), mode='nearest')

                # calculate the loss
                loss = criterion(output, masks, temperature = 1)

                # update the loss 
                self.update_loss(loss)

                # calculate the metrics
                metrics, confusion_matrix = eval_metrics(output, masks, self.num_classes)

                # udpate the metrics
                self.update_metrics(*metrics, confusion_matrix = confusion_matrix)

                # backward + optimize only if in training phase
                if phase == 'train':
                    # propagate the losses
                    loss.backward()
                    optimizer.step()

                # calculate the batch acc and iou
                pixel_acc = 1.0 * metrics[0] / (np.spacing(1) + metrics[1])
                IoU = 1.0 * metrics[2] / (np.spacing(1) + metrics[3])
                dice = (2 * IoU) / (IoU + 1)

                mIoU = IoU.mean()
                mDice = dice.mean()
                
                class_iou = dict(zip(self.class_names, np.round(IoU, 2))),
                class_dice = dict(zip(self.class_names, np.round(dice, 2)))


                # calculate the individual dice only
                for i_idx, pred in enumerate(output):
                    i_metric, _ = metrics, confusion_matrix = eval_metrics(pred[None,:], masks[i_idx][None,:], self.num_classes)
                    IoU = 1.0 * i_metric[2] / (np.spacing(1) + i_metric[3])
                    dice = (2 * IoU) / (IoU + 1)
                    all_f1_scores.append(dice.mean())

                pbar.set_description(f'Inner Loop - {phase:<5}: Epoch:{epoch} | Loss:{loss.item():.4f}'
                                     f' | mIOU:{mIoU:.2f}'
                                     f' | mDice:{mDice:.2f}'
                                     f' | Accuracy:{pixel_acc:.2f}'
                                     f' | IOU:{class_iou}'
                                     f' | Dice:{class_dice}')
                # update the counter
                pbar.update(1)

        # reset the counter to be used again
        pbar.reset()

        if phase == 'train':
            scheduler.step()

        # get epoch based metrics
        self.pixel_acc, self.mIoU, self.mDice, self.class_iou, self.class_dice, self.confusion_matrix = self.get_metrics().values()
        epoch_loss = self.loss.average
        epoch_accuracy = self.pixel_acc
        epoch_iou = self.mIoU
        epoch_f1 = self.mDice
        epoch_class_iou = self.class_iou
        epoch_class_dice = self.class_dice

        # log the scaler values here to tensorboard
        self.writer.add_scalar(f'{phase}-loss', epoch_loss, epoch)
        self.writer.add_scalar(f'{phase}-accuracy', epoch_accuracy, epoch)
        self.writer.add_scalar(f'{phase}-iou', epoch_iou, epoch)
        self.writer.add_scalar(f'{phase}-f1', epoch_f1, epoch)


        # log every ten epochs
        if epoch % 2 == 0:

            # log the confusion matrix
            cm_fig = plot_confusion_matrix(self.confusion_matrix, self.class_names)
            self.writer.add_figure(f'{phase}-confusion_matrix', cm_fig, epoch)
            plt.close()
            
            # log the images with predictions from random batch
            num_of_samples = 12
        
            # Get a random sample
            pbar.set_description(f'Generating Random performing predictions ...')
            random.seed(None)
            random_index = random.sample(range(len(generator.dataset)), k=num_of_samples)
            self.generate_predictions_figure(epoch, generator, model, num_of_samples, f'{phase}-random', random_index)

            # sort the array
            sorted_indices = np.argsort(all_f1_scores)
        
            # get the top instances
            pbar.set_description(f'Generating Top performing predictions ...')
            self.generate_predictions_figure(epoch, generator, model, num_of_samples, f'{phase}-top', sorted_indices[-num_of_samples:])
        
            # get the worst instances
            pbar.set_description(f'Generating Worst performing predictions ...')
            self.generate_predictions_figure(epoch, generator, model, num_of_samples, f'{phase}-worst', sorted_indices[:num_of_samples])

        return epoch_loss, epoch_accuracy, epoch_iou, epoch_f1, epoch_class_iou, epoch_class_dice

    def generate_predictions_figure(self, epoch, generator, model, num_of_samples, title, index):
        """
        Generates the figure with predictions and plots in the tensorboard

        :param epoch:
        :param generator:
        :param model:
        :param num_of_samples:
        :param title:
        :param index:
        """
        images = []
        masks = []
        names = []

        for idx in index:
            image, mask, name = generator.dataset[idx]
            images.append(image.numpy())
            masks.append(mask.numpy())
            names.append(name)

        images = np.array(images)
        masks = np.array(masks)
        names = np.array(names)

        with torch.set_grad_enabled(False):
            outputs = model(torch.from_numpy(images).cuda())
            outputs = F.interpolate(outputs, size = (self.patch_size, self.patch_size), mode='nearest')
            outputs = torch.nn.Softmax2d()(outputs)
            outputs = outputs.cpu().numpy()

        pred_fig = plot_image_prediction(images, masks, outputs, names, num_of_samples, self.num_classes)
        self.writer.add_figure(title, pred_fig, epoch)
        plt.close()

