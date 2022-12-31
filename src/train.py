from training.train_baseline import Trainer
import argparse
import config
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def main(args):

    baseline_trainer = Trainer(args['model'],
                                     args['data_path'],
                                     args['mini_patch_size'],
                                     args['input_size'],
                                     args['input_mag'],
                                     args['output_path'],
                                     args['num_classes'],
                                     args['class_names'],
                                     args['num_epochs'],
                                     args['batch_size'],
                                     args['gpu_id'],
                                     EXPERIMENT_NO)

    baseline_trainer.train_model()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description="Baseline trainer for Coarse Segmentation",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    MODEL = config.NORMAL_RESNET_MODEL
    DATA_PATH = '/mnt/sda2/coarse_segmentation/data/patches/augmented_coarse_wise/'
    INPUT_SIZE = 512
    MINI_PATCH_SIZE = 16
    NUM_CLASSES = config.TNBC_NUMBER_OF_CLASSES
    INPUT_MAGNIFICATION = config.PATCHES_MAGNIFICATION_LEVEL
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    DATA_NAME = 'TNBC'
    GPU = 0
    OUTPUT_PATH = '/mnt/sda2/coarse_segmentation/model_output/'
    CLASS_NAMES = ['BG', 'Tumor','Stroma','Inflam', 'Dead']

    EXPERIMENT_NO = f'd{DATA_NAME}_p{INPUT_SIZE}_z{INPUT_MAGNIFICATION}_c{NUM_CLASSES}_mp{MINI_PATCH_SIZE}_m{MODEL}_RandAugment'

    parser.add_argument("--data_path", default=DATA_PATH, help="Path to data folder location")
    parser.add_argument("--model", default=MODEL, help="Model name where there are following models: unet, deeplab, segnet, fcn etc. by default its unet")
    parser.add_argument("--input_size", default=INPUT_SIZE, help="Network input size by default its 512")
    parser.add_argument("--input_mag", default=INPUT_MAGNIFICATION, help="Network input magnification [10, 20, 40] by default its 20 ")
    parser.add_argument("--num_classes", default=NUM_CLASSES, help="Number of classes in the dataset")
    parser.add_argument("--batch_size", default=BATCH_SIZE, help="Batch size where by default its 8")
    parser.add_argument("--num_epochs", default=NUM_EPOCHS, help="Number of epochs by default its 100")
    parser.add_argument("--data_name", default=DATA_NAME, help="Dataset name e.g., tnbc")
    parser.add_argument("--output_path", default=OUTPUT_PATH, help="Output data folder location")
    parser.add_argument("--gpu_id", default=GPU, type=int, help="gpu id")
    parser.add_argument("--mini_patch_size", default=MINI_PATCH_SIZE, help="Mini patch size [1, 8, 16, 32]")
    parser.add_argument("--class_names", default=CLASS_NAMES, help="Array of class names")

    args = vars(parser.parse_args())

    main(args)

    
