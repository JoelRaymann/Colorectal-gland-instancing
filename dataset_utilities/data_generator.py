# Import the packages
import numpy as np
import tensorflow as tf
import yaml
from PIL import Image
from albumentations import Compose
from albumentations import Flip, ShiftScaleRotate, RandomCrop
from albumentations import HueSaturationValue, RandomBrightnessContrast
from albumentations import MedianBlur, GaussianBlur, GaussNoise, MotionBlur, ElasticTransform
from cv2 import cv2 as cv


# Define Class Data Generator with Augmentation
class DataGenerator(tf.keras.utils.Sequence):

    # NOTE: Override in-built functions
    def __init__(self, dataset_path_set: list, image_size: tuple, resize: int, model_name: str, batch_size=32, aug_config_path=None,
                 enable_patching = False, apply_augmentation=True, shuffle=True):
        """
        A Class for setting up keras data generator.

        :param dataset_path_set: The list of tuples consisting of (image, map, overlay) paths.
        :param image_size: A tuple mentioning the size of the images as (W, H)
        :param resize: A integer to resize the image.
        :param model_name: The name of the model to be trained - {"UNet", "MIMO-Net", "UNet-Attention"}
        :param batch_size: the size of the batch to process (default: 32)
        :param aug_config_path: The configuration yaml path for the augmentation (default: None)
        :param enable_patching: A status flag to use patch extraction instead of resize. (default: False)
        :param apply_augmentation: The status flag for enabling augmentation (default: True)
        :param shuffle: The status flag for enabling shuffling (default: True)
        """
        np.random.seed(100)
        self.dataset_path_set = dataset_path_set
        self.image_size = image_size
        self.resize = resize
        self.model_name = model_name
        self.batch_size = batch_size
        self.enable_patching = enable_patching
        self.apply_augmentation = apply_augmentation
        self.shuffle = shuffle
        self.AUGMENTATION_CONFIG_PATH = aug_config_path
        self.__setup_augmentation()
        # Call a epoch end for starting a shuffle if necessary
        self.on_epoch_end()

    # NOTE: Implement Core Functions
    def __read_augmentation_config(self) -> dict:
        """
        Function to read the yaml configuration file
        and return the dictionary
        :return: dict
        """
        if self.AUGMENTATION_CONFIG_PATH is None:
            print("[ERROR]: Augmentation Configuration Path not Mentioned!")
            raise FileNotFoundError

        else:
            return yaml.load(open(self.AUGMENTATION_CONFIG_PATH))

    def __setup_augmentation(self):
        """
        Function to setup the augmentation pipeline
        based on the configuration
        :return: None
        """
        aug_config = self.__read_augmentation_config()
        # Retrieve Configurations
        HSV = aug_config["HSV_Saturation"]
        RBC = aug_config["Random_Brightness_Contrast"]
        Flip_aug = aug_config["Flip"]
        Linear_aug = aug_config["Linear_Augmentation"]
        Median_Blur = aug_config["Median_Blur"]
        Gaussian_Blur = aug_config["Gaussian_Blur"]
        Gaussian_Noise = aug_config["Gaussian_Noise"]
        Motion_Blur = aug_config["Motion_Blur"]
        Elastic_Blur = aug_config["Elastic_Blur"]

        # Load up functions
        self.aug_fn = Compose([
            HueSaturationValue(hue_shift_limit=HSV["hue_shift_limit"], sat_shift_limit=HSV["sat_shift_limit"],
                               val_shift_limit=HSV["val_shift_limit"], always_apply=HSV["always_apply"], p=HSV["p"]),
            RandomBrightnessContrast(brightness_limit=RBC["brightness_limit"], contrast_limit=RBC["contrast_limit"],
                                     brightness_by_max=RBC["brightness_by_max"], always_apply=RBC["always_apply"],
                                     p=RBC["p"]),
            MedianBlur(Median_Blur["blur_limit"], Median_Blur["always_apply"], Median_Blur["p"]),
            GaussianBlur(Gaussian_Blur["blur_limit"], Gaussian_Blur["always_apply"], Gaussian_Blur["p"]),
            GaussNoise((Gaussian_Noise["var_limit_low"], Gaussian_Noise["var_limit_high"]), Gaussian_Noise["mean"],
                       Gaussian_Noise["always_apply"], Gaussian_Noise["p"]),
            MotionBlur(Motion_Blur["blur_limit"], Motion_Blur["always_apply"], Motion_Blur["p"]),
        ])

        self.aug_fn_all = Compose([
            Flip(always_apply=Flip_aug["always_apply"], p=Flip_aug["p"]),
            ShiftScaleRotate(always_apply=Linear_aug["always_apply"], p=Linear_aug["p"]),
            RandomCrop(height=self.image_size[1], width=self.image_size[0], always_apply=True, p=1.0),
            ElasticTransform(interpolation=cv.INTER_AREA, always_apply=Elastic_Blur["always_apply"],
                             p=Elastic_Blur["p"])
        ])

        self.patch_extractor = Compose([
            RandomCrop(height = self.resize, width = self.resize, always_apply = True, p = 1.0)
        ])

        print("[INFO]: Loaded the Augmentation Functions based on the configuration.")
        print("[INFO]: To apply a change, call reload_augmentation()")
        return None

    def reload_augmentation(self, augmentation_config_file: str):
        """
        Reloads the augmentation functions based on the new
        configuration file
        :param augmentation_config_file: The path to the new config file
        :return: None
        """
        self.AUGMENTATION_CONFIG_PATH = augmentation_config_file
        print("[INFO]: Generating new augmentation functions")
        self.__setup_augmentation()
        return None

    def __load_data_mimonet(self, dataset_paths: list) -> tuple:
        """
        Function to load the images from the given indexes, resize
        them to resize x resize using inter_area interpolation, apply augmentation,
        and return the tuple of (image, map).

        :param dataset_paths: The range of indexes to load and work
        :return: tuple of (image, map)
        """

        # Initialize
        X = np.empty((self.batch_size, *self.image_size, 3), dtype = "float32")
        X1 = np.empty((self.batch_size, int(self.image_size[1] / 2), int(self.image_size[0] / 2), 3), dtype="float32")
        X2 = np.empty((self.batch_size, int(self.image_size[1] / 4), int(self.image_size[0] / 4), 3), dtype="float32")
        X3 = np.empty((self.batch_size, int(self.image_size[1] / 8), int(self.image_size[0] / 8), 3), dtype="float32")
        X4 = np.empty((self.batch_size, int(self.image_size[1] / 16), int(self.image_size[0] / 16), 3), dtype="float32")
        y = np.empty((self.batch_size, *self.image_size), dtype = "float32")

        for index, paths in enumerate(dataset_paths):
            # Read the images
            img = Image.open(paths[0])
            img = np.copy(np.asarray(img))

            # Read and correct the map
            seg = Image.open(paths[1])
            seg = np.copy(np.asarray(seg))
            seg[seg > 0.0] = 255
            seg[seg == 0.0] = 0

            if self.apply_augmentation is True:

                if self.enable_patching:
                    # Extract random resize x resize patch
                    patch = self.patch_extractor(image = img, mask = seg)
                    img = patch["image"]
                    seg = patch["mask"]
                else:
                    # resize the images
                    img = cv.resize(img, (self.resize, self.resize), interpolation=cv.INTER_AREA)
                    seg = cv.resize(seg, (self.resize, self.resize), interpolation=cv.INTER_AREA)

                # Apply the augmentation
                transformed = self.aug_fn(image=img)
                img = transformed["image"]

                transformed = self.aug_fn_all(image=img, mask=seg)
                img = transformed["image"]
                seg = transformed["mask"]

            else:
                img = cv.resize(img, (self.image_size[1], self.image_size[0]), interpolation = cv.INTER_AREA)
                seg = cv.resize(seg, (self.image_size[1], self.image_size[0]), interpolation = cv.INTER_AREA)

            # Resize and get all inputs
            img1 = cv.resize(img, (int(self.image_size[1] / 2), int(self.image_size[0] / 2)), interpolation = cv.INTER_AREA)
            img2 = cv.resize(img, (int(self.image_size[1] / 4), int(self.image_size[0] / 4)), interpolation = cv.INTER_AREA)
            img3 = cv.resize(img, (int(self.image_size[1] / 8), int(self.image_size[0] / 8)), interpolation = cv.INTER_AREA)
            img4 = cv.resize(img, (int(self.image_size[1] / 16), int(self.image_size[0] / 16)), interpolation = cv.INTER_AREA)

            # Normalize it b/n 0 - 1
            img = img / 255.0
            img1 = img1 / 255.0
            img2 = img2 / 255.0
            img3 = img3 / 255.0
            img4 = img4 / 255.0
            seg = seg / 255.0

            # Fill the data
            X[index,] = img
            X1[index,] = img1
            X2[index,] = img2
            X3[index,] = img3
            X4[index,] = img4
            y[index,] = seg

        return [X, X1, X2, X3, X4], tf.expand_dims(y, axis=-1)

    def __load_data_unet(self, dataset_paths: list) -> tuple:
        """
        Function to load the images from the given indexes, resize
        them to resize x resize using inter_area interpolation, apply augmentation,
        and return the tuple of (image, map).

        :param dataset_paths: The range of indexes to load and work
        :return: tuple of (image, map)
        """

        # Initialize
        X = np.empty((self.batch_size, *self.image_size, 3), dtype = "float32")
        y = np.empty((self.batch_size, *self.image_size), dtype = "float32")

        for index, paths in enumerate(dataset_paths):
            # Read the images
            img = Image.open(paths[0])
            img = np.copy(np.asarray(img))

            # Read and correct the map
            seg = Image.open(paths[1])
            seg = np.copy(np.asarray(seg))
            seg[seg > 0.0] = 255
            seg[seg == 0.0] = 0

            if self.apply_augmentation is True:

                if self.enable_patching:
                    # Extract random resize x resize patch
                    patch = self.patch_extractor(image = img, mask = seg)
                    img = patch["image"]
                    seg = patch["mask"]
                else:
                    # resize the images
                    img = cv.resize(img, (self.resize, self.resize), interpolation=cv.INTER_AREA)
                    seg = cv.resize(seg, (self.resize, self.resize), interpolation=cv.INTER_AREA)

                # Apply the augmentation
                transformed = self.aug_fn(image=img)
                img = transformed["image"]

                transformed = self.aug_fn_all(image=img, mask=seg)
                img = transformed["image"]
                seg = transformed["mask"]

            else:
                img = cv.resize(img, (self.image_size[1], self.image_size[0]), interpolation = cv.INTER_AREA)
                seg = cv.resize(seg, (self.image_size[1], self.image_size[0]), interpolation = cv.INTER_AREA)

            # Normalize it b/n 0 - 1
            img = img / 255.0
            seg = seg / 255.0

            # Fill the data
            X[index,] = img
            y[index,] = seg

        return X, tf.expand_dims(y, axis=-1)

    # NOTE: Implement necessary functions
    def __len__(self) -> int:
        """
        Denotes the number of batches per epoch
        :return: The no. of batches
        """
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index) -> tuple:
        """
        Function to generate the item (X,y) for
        training
        :param index: the index of the generator
        :return: A tuple consisting of (X, y)
        """
        # Generate the indexes of the batch
        while index >= self.__len__():
            index -= self.__len__()

        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Get the dataset_path for the indexes
        dataset_paths = [self.dataset_path_set[element] for element in indexes]

        # Generate the dataset
        if self.model_name == "UNet" or self.model_name == "UNet-Attention":
            inp, out = self.__load_data_unet(dataset_paths=dataset_paths)

        elif self.model_name == "MIMO-Net":
            return self.__load_data_mimonet(dataset_paths=dataset_paths)
        else:
            raise Exception

        return inp, out

    def on_epoch_end(self):
        """
        Update the indexes for shuffling if necessary after an epoch
        :return: None
        """
        self.indexes = np.arange(len(self.dataset_path_set))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        return None