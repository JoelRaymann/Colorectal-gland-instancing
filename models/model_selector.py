# Implement model selector

# Import necessary packages
import tensorflow as tf

from .MIMONet import MIMO_Net
# Import in-house models
from .UNet import UNet
from .UNetAttention import UNetAttention


def ModelSelector(model_name: str, input_size = (464, 464, 3)) -> tf.keras.Model:
    """
    Function to select and build the model of your choice for training and testing.
    This function returns the built version of the requested model for compilation
    and training/testing.

    Models to select from:
        "UNet" : The standard U-Net model

    :param model_name: The model name to build and train
    :param input_size: A tuple of input size in dimension == (width, height, channel)

    :return: the tf.keras.Model for training and testing NOTE: Need Compilation
    """

    print("[INFO]: Selecting model: {0} with input size: {1}".format(model_name, input_size))

    if model_name == "UNet":
        return UNet(input_size)

    elif model_name == "UNet-Attention":
        return UNetAttention(input_size)

    elif model_name == "MIMO-Net":
        return MIMO_Net(input_size)

    else:
        raise Exception
    # More models incoming