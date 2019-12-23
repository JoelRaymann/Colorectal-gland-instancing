# Implement morphological operations in here for post-processing

import numpy as np
# import necessary functions
import yaml
from cv2 import cv2 as cv


# develop custom post-processing
class PostProcessing:
    """
    A class for applying postprocessing operations
    before evaluation, set postprocessing_config.yaml
    file before using this
    """
    def __init__(self, config_path = "./postprocessing_config.yaml"):
        """
        A Class that sets the postprocessing functions using
        the configuration in the postprocessing_config.yaml.

        :param config_path: The path for the postprocessing_config.yaml
        """
        self.POSTPROCESSING_CONFIG = config_path
        self.__load_config()
        print(self)

    # Implement Core Functionalities
    def __setup_postprocessing(self):
        """
        Function to open the yaml file and
        set it up
        :return: a dictionary of the yaml
        configurations
        """
        return yaml.load(open(self.POSTPROCESSING_CONFIG))

    def __load_config(self) :
        """
        Function to load the configuration and set the functions
        in order as a list.
        """
        # Get the configuration
        config = self.__setup_postprocessing()
        # collect the enabled configurations
        enabled_config = {k: v for k, v in config.items() if v["enabled"] is True}
        # Sort the functions in order
        self.enabled_config = {k: v for k, v in sorted(enabled_config.items(), key = lambda item: item[1]["order"])}

    def __apply_postprocessing(self, image, kernel = None):
        """
        Function to apply the postprocessing functions
        in order

        :param image: The image to post-process
        :param kernel: The custom kernel to apply.
        Use this only if any one of the functions were
        set

        :return: The post-processed image of the input
        """
        disk_filter = np.array([
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 0, 1],
        ], dtype = "uint8")

        for key, value in self.enabled_config.items():

            if key == "Erosion":
                if value["filter"] is "disk":
                    image = cv.erode(image, disk_filter, value["iterations"])
                else:
                    image = cv.erode(image, kernel, value["iterations"])

            if key == "Dilation":
                if value["filter"] is "disk":
                    image = cv.dilate(image, disk_filter, value["iterations"])
                else:
                    image = cv.dilate(image, kernel, value["iterations"])

            if key == "Opening":
                if value["filter"] is "disk":
                    image = cv.morphologyEx(image, cv.MORPH_OPEN, disk_filter, iterations = value["iterations"])
                else:
                    image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations = value["iterations"])

            if key == "Closing":
                if value["filter"] is "disk":
                    image = cv.morphologyEx(image, cv.MORPH_CLOSE, disk_filter, iterations = value["iterations"])
                else:
                    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations = value["iterations"])

        return image

    def apply(self, image, kernel = None):
        """
        Function to apply the postprocessing functions
        in order

        :param image: The image to post-process
        :param kernel: The custom kernel to apply.
        Use this only if any one of the functions were
        set

        :return: The post-processed image of the input
        """
        return self.__apply_postprocessing(image, kernel)

    def __str__(self):
        string = "Post-processing Functions enabled in order\n"

        for index, key in enumerate(self.enabled_config):
            string += "[INFO]: {0} => {1}\n".format(index, key)

        return string
