# Import necessary packages
import tensorflow as tf

import layers


def UNetAttention(input_size=(464, 464, 3)):
    """
    Function to build and return the Attention based
    U-Net models with the given input_size

    :param input_size: A tuple consisting of the input dimensions eg. (464, 464, 3)
    :return: A tf.keras.Model of the U-Net which can be compiled and trained or loaded
    with weights and then trained - y'know the usual stuffs.
    """
    X = tf.keras.layers.Input(shape=input_size, dtype="float32", name="input_layer")

    # Level - 1 down
    conv1 = tf.keras.layers.Conv2D(64, 3, activation="elu", padding="same", name="L1_conv1")(X)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation="elu", padding="same", name="L1_conv2")(conv1)
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="L1_pool")(conv1)

    # Level - 2 down
    conv2 = tf.keras.layers.Conv2D(128, 3, activation="elu", padding="same", name="L2_conv1")(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation="elu", padding="same", name="L2_conv2")(conv2)
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="L2_pool")(conv2)

    # Level - 3 down
    conv3 = tf.keras.layers.Conv2D(256, 3, activation="elu", padding="same", name="L3_conv1")(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation="elu", padding="same", name="L3_conv2")(conv3)
    pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="L3_pool")(conv3)

    # Level - 4 down
    conv4 = tf.keras.layers.Conv2D(512, 3, activation="elu", padding="same", name="L4_conv1")(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation="elu", padding="same", name="L4_conv2")(conv4)
    # drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="L4_pool")(conv4)

    # Bottleneck layers
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation="elu", padding="same", name="L5_conv1")(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation="elu", padding="same", name="L5_conv2")(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    # Level - 4 up
    up4 = tf.keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), activation="elu", name="U4_convT1")(drop5)
    attn4, attn4_map = layers.AttentionLayer(attn_unit=512, name="U4_attn")(conv4, up4)
    merge4 = tf.keras.layers.Concatenate(axis=-1, name="U4_merge")([attn4, up4])
    up4 = tf.keras.layers.Conv2D(512, 3, activation="elu", padding="same", name="U4_conv1")(merge4)
    up4 = tf.keras.layers.Conv2D(512, 3, activation="elu", padding="same", name="U4_conv2")(up4)

    # Level - 3 up
    up3 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), activation="elu", name="U3_convT1")(up4)
    attn3, attn3_map = layers.AttentionLayer(attn_unit=256, name="U3_attn")(conv3, up3)
    merge3 = tf.keras.layers.Concatenate(axis=-1, name="U3_merge")([attn3, up3])
    up3 = tf.keras.layers.Conv2D(256, 3, activation="elu", padding="same", name="U3_conv1")(merge3)
    up3 = tf.keras.layers.Conv2D(256, 3, activation="elu", padding="same", name="U3_conv2")(up3)

    # Level - 2 up
    up2 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), activation="elu", name="U2_convT1")(up3)
    attn2, attn2_map = layers.AttentionLayer(attn_unit=128, name="U2_attn")(conv2, up2)
    merge2 = tf.keras.layers.Concatenate(axis=-1, name="U2_merge")([attn2, up2])
    up2 = tf.keras.layers.Conv2D(128, 3, activation="elu", padding="same", name="U2_conv1")(merge2)
    up2 = tf.keras.layers.Conv2D(128, 3, activation="elu", padding="same", name="U2_conv2")(up2)

    # Level - 1 up
    up1 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), activation="elu", name="U1_convT1")(up2)
    attn1, attn1_map = layers.AttentionLayer(attn_unit=64, name="U1_attn")(conv1, up1)
    merge1 = tf.keras.layers.Concatenate(axis=-1, name="U1_merge")([attn1, up1])
    up1 = tf.keras.layers.Conv2D(64, 3, activation="elu", padding="same", name="U1_conv1")(merge1)
    up1 = tf.keras.layers.Conv2D(64, 3, activation="elu", padding="same", name="U1_conv2")(up1)

    # output layers
    drop_y = tf.keras.layers.Dropout(0.5, name="output_drop")(up1)
    y = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", padding="same", name="output_layer")(drop_y)

    return tf.keras.Model(inputs=X, outputs=y)