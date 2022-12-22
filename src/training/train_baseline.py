import argparse
import copy
import os
import random
import shutil
import albumentations as album
import numpy as np
import torch
import torchsummary
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.data_reader import DataReader
import config
from utils.losses import *
from utils.metrics import *


class TrainBaseline:
    """
    Training the baseline coarse segmentation networks

    """

    def __init__(self, model_name, data_path, semi_split_divisor, patch_size, patch_magnification, output_path, num_classes, num_epochs, batch_size, gpu_id, experiment_no):
        self.model_name = model_name
        self.data_path = data_path
        self.semi_split_divisor = semi_split_divisor
        self.patch_size = patch_size
        self.patch_magnification = patch_magnification
        self.output_path = output_path
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.experiment_no = experiment_no
        self.patch_folder = 'norm_patch' if 'norm' in self.experiment_no else 'patch'
        self.config_no = self.experiment_no.split('_')[-1]

        # create directories path
        self.log_dir = os.path.join(self.output_path, 'logs', self.experiment_no)
        self.tensorboard_dir = os.path.join(self.output_path, 'tensorboard', self.experiment_no)
        self.models_dir = os.path.join(self.output_path, 'models', self.experiment_no)
        self.raw_dir = os.path.join(self.output_path, 'raw', self.experiment_no)

        # check for debug
        if 'debug' in self.experiment_no and os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir, ignore_errors=True)
            shutil.rmtree(self.models_dir, ignore_errors=True)
            shutil.rmtree(self.raw_dir, ignore_errors=True)
            shutil.rmtree(self.tensorboard_dir, ignore_errors=True)

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

    def load_model(self, model_name):
        '''
        load the semantic semgentation models for training the baseline
        :return: model
        '''

        encoder = 'efficientnet-b0'
        encoder_weights = 'imagenet'
        activation = 'softmax2d'  # could be None for logits or 'softmax2d' for multiclass segmentation
        model = None

        # create segmentation model with pretrained encoder
        if model_name == 'unet':
            """
            Unet
            """
            model = smp.Unet(
                    encoder_name=encoder,
                    encoder_weights=encoder_weights,
                    classes=self.num_classes,
                    activation=activation)

        elif model_name == 'deeplab':
            """
            DeepLabV3+
            """
            model = smp.DeepLabV3Plus(
                    encoder_name=encoder,
                    encoder_weights=encoder_weights,
                    classes=self.num_classes,
                    activation=activation)

        elif model_name == 'fpn':
            """
            FPN
            """
            model = smp.FPN(
                    encoder_name=encoder,
                    encoder_weights=encoder_weights,
                    classes=self.num_classes,
                    activation=activation)

        elif model_name == 'psp':
            """
            PSP Net
            """
            model = smp.PSPNet(
                    encoder_name=encoder,
                    encoder_weights=encoder_weights,
                    classes=self.num_classes,
                    activation=activation)

        else:

            model = smp.Unet(
                    encoder_name=encoder,
                    encoder_weights=encoder_weights,
                    classes=self.num_classes,
                    activation=activation)

        # put model to the GPU
        model.cuda()

        # print the network
        torchsummary.summary(model, (3, self.patch_size, self.patch_size))

        return model

    def load_data(self):
        """
        load the data and returns the train and test generators

        :return: train and test generators
        """

        self.logger.info(f'STEP:2 - Creating augmentation for train and test')
        # create the train augmentation
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
        train_patch_path = os.path.join(self.data_path, self.patch_folder, f'{self.patch_magnification}x', f'{self.patch_size}', 'train')
        test_patch_path = os.path.join(self.data_path, self.patch_folder, f'{self.patch_magnification}x', f'{self.patch_size}', 'test')

        # get the config files for filtering the images
        train_config_file_path = os.path.join(self.data_path, f'config_{self.config_no}', f'supervised_1by{self.semi_split_divisor}_list.npy')
        test_config_file_path = os.path.join(self.data_path, f'config_{self.config_no}', f'test_1by{self.semi_split_divisor}_list.npy')

        # load the config files
        config_file = np.load(train_config_file_path)
        config_test_file = np.load(test_config_file_path)
        self.logger.info(f'STEP:2 - Loading the configuration files from \ntrain: {train_config_file_path} \ntest: {test_config_file_path}')

        # create dataset reader
        data_reader = TNBCDataset(data_dir=train_patch_path, num_classes=self.num_classes, target_size=self.patch_size, config_file=config_file, transformation=train_transform)
        data_test_reader = TNBCDataset(data_dir=test_patch_path, num_classes=self.num_classes, target_size=self.patch_size, config_file=config_test_file, transformation=test_transform)
        self.logger.info(f'STEP:2 - Loading the data files from \ntrain:{train_patch_path} \ntest: {test_patch_path}')

        # create the data generator for pytorch
        train_generator = torch.utils.data.DataLoader(data_reader, **params)
        test_generator = torch.utils.data.DataLoader(data_test_reader, **params)

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

    def train_model(self):
        """
        train loop for training the model which includes creating the model,
        reading data and training the model using the data

        """
        self.logger.info(f'STEP:1 - Loading model {self.model_name}')
        # load the model
        model = self.load_model(self.model_name)
        self.logger.info(f'STEP:1 - Model {self.model_name} loaded')

        self.logger.info(f'STEP:2 - Loading data generators ')
        # load the data generators
        train_generator, test_generator = self.load_data()
        self.logger.info(f'STEP:2 - Data generators loaded')

        self.logger.info(f'STEP:3 - Loading optimizers')
        optim = self.get_optim(model, 'adam', 0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=0.00001)
        self.logger.info(f'STEP:3 - Loaded \n{optim}')

        self.logger.info(f'STEP:4 - Loading losses')
        # losses = [smp.utils.losses.DiceLoss(),
        #           smp.utils.losses.CrossEntropyLoss(),
        #           smp.utils.losses.JaccardLoss()]
        losses = smp.losses.DiceLoss(mode='multilabel', from_logits=False)

        self.logger.info(f'STEP:5 - Starting the training loop')

        # progress bar outer for epoch
        pbar_out_train = tqdm(total=self.num_epochs, position=0)
        pbar_out_test = tqdm(total=self.num_epochs, position=2)

        # progress bar inner for epoch batches
        pbar_train = tqdm(total=len(train_generator), position=1)
        pbar_test = tqdm(total=len(test_generator), position=3)

        best_f1 = 0
        best_model = None
        best_epoch = 0

        # put 0's

        # loop through the epochs
        for epoch in range(self.num_epochs):

            phase = 'train'
            # run the train step
            epoch_loss, epoch_accuracy, epoch_iou, epoch_f1, epoch_f1_beta = self.model_step(epoch, phase, model, train_generator, optim, losses, scheduler, pbar_train)

            # save the best model
            if best_f1 < epoch_f1:
                best_f1 = epoch_f1
                best_model = copy.deepcopy(model)
                best_epoch = epoch

            train_string = f'Main Loop  - {phase}: Epoch:{epoch}' \
                           f' | Loss:{epoch_loss:.4f}' \
                           f' | IOU:{epoch_iou:.4f}' \
                           f' | F1:{epoch_f1:.4f}' \
                           f' | F1_beta:{epoch_f1_beta:.4f}' \
                           f' | Accuracy:{epoch_accuracy:.4f}'
            pbar_out_train.set_description(train_string)

            phase = 'test'
            # run the test step
            epoch_loss, epoch_accuracy, epoch_iou, epoch_f1, epoch_f1_beta = self.model_step(epoch, phase, model, test_generator, optim, losses, scheduler, pbar_test)

            test_string = f'Main Loop  - {phase}:  Epoch:{epoch}' \
                          f' | Loss:{epoch_loss:.4f}' \
                          f' | IOU:{epoch_iou:.4f}' \
                          f' | F1:{epoch_f1:.4f}' \
                          f' | F1_beta:{epoch_f1_beta:.4f}' \
                          f' | Accuracy:{epoch_accuracy:.4f}'
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
        self.model_step(epoch + 1, phase, best_model, train_generator, optim, losses, scheduler, pbar_train)

        phase = 'test'
        # run the test step
        self.model_step(epoch + 1, phase, best_model, test_generator, optim, losses, scheduler, pbar_test)

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
        if phase == 'train':
            # set the model in training mode
            model.train()
        else:
            model.eval()

        # running loss for the training
        running_loss = 0.0
        running_accuracy = 0.0
        running_f1 = 0.0
        running_f1_beta = 0.0
        running_iou = 0.0

        # used to get top,worst and random images
        all_f1_scores = []

        # load the batch
        for images, masks, _ in generator:

            # move the batch to cuda device as the model is on cuda
            images = images.to(self.device)
            masks = masks.to(self.device)

            # permute the masks to match the output
            masks = masks.permute(0, 3, 1, 2)

            # reset the optimizer gradients to zero for next epoch calculations
            optimizer.zero_grad()

            # for safe check make sure gradients are enabled by working in enabled scope
            with torch.set_grad_enabled(phase == 'train'):

                # pass the images through the model and get output
                output = model(images)

                # calculate the loss
                loss = criterion(output, masks)

                tp, fp, fn, tn = smp.metrics.get_stats(output, masks, mode='multilabel', threshold=0.5)

                # then compute metrics with required reduction (see metric docs)
                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
                accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
                recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

                # backward + optimize only if in training phase
                if phase == 'train':
                    # propagate the losses
                    loss.backward()
                    optimizer.step()

                # accumulate loss
                running_loss += loss.item()
                running_iou += iou_score
                running_f1 += f1_score
                running_f1_beta += f2_score
                running_accuracy += accuracy

                # calculate the individual f1 only
                for i_idx, pred in enumerate(output):
                    i_tp, i_fp, i_fn, i_tn = smp.metrics.get_stats(pred, masks[i_idx], mode='multilabel', threshold=0.5)

                    all_f1_scores.append(smp.metrics.f1_score(i_tp, i_fp, i_fn, i_tn, reduction="micro").cpu().numpy())

                pbar.set_description(f'Inner Loop - {phase:<5}: Epoch:{epoch} | Loss:{loss.item():.4f}'
                                     f' | IOU:{iou_score:.4f}'
                                     f' | F1:{f1_score:.4f}'
                                     f' | F1_beta:{f2_score:.4f}'
                                     f' | Accuracy:{accuracy:.4f}'
                                     f' | Recall:{recall:.4f}')
                # update the counter
                pbar.update(1)

        # reset the counter to be used again
        pbar.reset()

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / len(generator)
        epoch_accuracy = running_accuracy / len(generator)
        epoch_iou = running_iou / len(generator)
        epoch_f1 = running_f1 / len(generator)
        epoch_f1_beta = running_f1_beta / len(generator)

        # log the scaler values here to tensorboard
        self.writer.add_scalar(f'{phase}-loss', epoch_loss, epoch)
        self.writer.add_scalar(f'{phase}-accuracy', epoch_accuracy, epoch)
        self.writer.add_scalar(f'{phase}-iou', epoch_iou, epoch)
        self.writer.add_scalar(f'{phase}-f1', epoch_f1, epoch)
        self.writer.add_scalar(f'{phase}-f1_beta', epoch_f1_beta, epoch)

        # log every ten epochs
        # if epoch % 10 == 0:
        #     # log the images with predictions from random batch
        #     num_of_samples = 12
        #
        #     # Get a random sample
        #     pbar.set_description(f'Generating Random performing predictions ...')
        #     random_index = random.sample(range(len(generator.dataset)), k=num_of_samples)
        #     self.generate_predictions_figure(epoch, generator, model, num_of_samples, f'{phase}-random', random_index)
        #
        #     # # sort the array
        #     sorted_indices = np.argsort(all_f1_scores)
        #
        #     # get the top instances
        #     pbar.set_description(f'Generating Top performing predictions ...')
        #     self.generate_predictions_figure(epoch, generator, model, num_of_samples, f'{phase}-top', sorted_indices[-num_of_samples:])
        #
        #     # get the worst instances
        #     pbar.set_description(f'Generating Worst performing predictions ...')
        #     self.generate_predictions_figure(epoch, generator, model, num_of_samples, f'{phase}-worst', sorted_indices[:num_of_samples])

        return epoch_loss, epoch_accuracy, epoch_iou, epoch_f1, epoch_f1_beta

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
            outputs = outputs.cpu().numpy()

        pred_fig = plot_image_prediction(images, masks, outputs, names, num_of_samples, self.num_classes)
        self.writer.add_figure(title, pred_fig, epoch)

        pred_fig.clf()
        plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description="Baseline trainer for Semantic Segmentation",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    MODEL = 'unext'
    DATA_PATH = '/home/saad/Desktop/temp_data/tnbc/'
    SEMI_SPLIT_DIVISOR = 1
    INPUT_SIZE = 512
    NUM_CLASSES = config.REDUCED_NUBER_OF_CLASSES
    INPUT_MAGNIFICATION = 10
    BATCH_SIZE = 10
    NUM_EPOCHS = 50
    DATA_NAME = 'TNBC'
    GPU = 0
    OUTPUT_PATH = '/mnt/sda2/tnbc_segmetnation_v2/output/tnbc/'
    EXPERIMENT_NO = f'm{MODEL}_d{DATA_NAME}_s{SEMI_SPLIT_DIVISOR}_p{INPUT_SIZE}_z{INPUT_MAGNIFICATION}_c{NUM_CLASSES}_baseline_aspp_1'

    parser.add_argument("--data_path", default=DATA_PATH, help="Path to data folder location")
    parser.add_argument("--model", default=MODEL, help="Model name where there are following models: unet, deeplab, segnet, fcn etc. by default its unet")
    parser.add_argument("--semi_split", default=SEMI_SPLIT_DIVISOR, help="Semi_supervised split [1, 2, 4, 8] by default its 1")
    parser.add_argument("--input_size", default=INPUT_SIZE, help="Network input size by default its 512")
    parser.add_argument("--input_mag", default=INPUT_MAGNIFICATION, help="Network input magnification [10, 20, 40] by default its 20 ")
    parser.add_argument("--num_classes", default=NUM_CLASSES, help="Number of classes in the dataset")
    parser.add_argument("--batch_size", default=BATCH_SIZE, help="Batch size where by default its 8")
    parser.add_argument("--num_epochs", default=NUM_EPOCHS, help="Number of epochs by default its 100")
    parser.add_argument("--data_name", default=DATA_NAME, help="Dataset name e.g., tnbc")
    parser.add_argument("--output_path", default=OUTPUT_PATH, help="Output data folder location")
    parser.add_argument("--gpu_id", default=GPU, type=int, help="gpu id")

    args = vars(parser.parse_args())

    baseline_trainer = TrainBaseline(args['model'],
                                     args['data_path'],
                                     args['semi_split'],
                                     args['input_size'],
                                     args['input_mag'],
                                     args['output_path'],
                                     args['num_classes'],
                                     args['num_epochs'],
                                     args['batch_size'],
                                     args['gpu_id'],
                                     EXPERIMENT_NO)

    baseline_trainer.train_model()
