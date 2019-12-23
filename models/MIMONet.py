# Implement MIMO-Net Here

# Import necessary packages
import tensorflow as tf

# OUR VARIANT OF MIMO-NET
def MIMO_Net(input_size = (256, 256, 3)):
    """
    Function to implement the MIMO-Net and provide the tf.keras.Model
    for compilation and training/testing

    :param input_size: The size of the input image
    :return: tf.keras.Model of the MIMO-Net
    """
    X = tf.keras.layers.Input(shape=input_size, dtype = "float32", name = "main_input")

    # DownSampling path
    # Group 1:
    # Block 1
    conv1 = tf.keras.layers.Conv2D(64, 3, activation="tanh", padding="same", name="conv1_block1")(X)
    conv2 = tf.keras.layers.Conv2D(64, 3, activation="tanh", padding="same", name="conv2_block1")(conv1)
    pool = tf.keras.layers.MaxPool2D((2, 2), name="pool_block1")(conv2)

    X1 = tf.keras.layers.Input(shape=(input_size[0] / 2, input_size[1] / 2, input_size[2]), dtype="float32",
                               name="downsampled_input1")
    conv3 = tf.keras.layers.Conv2D(64, 3, activation="tanh", padding="same", name="conv3_block1")(X1)
    conv4 = tf.keras.layers.Conv2D(64, 3, activation="tanh", padding="same", name="conv4_block1")(conv3)
    concat_block1 = tf.keras.layers.Concatenate(axis=-1, name="block1_concat")([pool, conv4])

    # Block 2
    conv1 = tf.keras.layers.Conv2D(128, 3, activation="tanh", padding="same", name="conv1_block2")(concat_block1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation="tanh", padding="same", name="conv2_block2")(conv1)
    pool = tf.keras.layers.MaxPool2D((2, 2), name="pool_block2")(conv2)

    X2 = tf.keras.layers.Input(shape=(input_size[0] / 4, input_size[1] / 4, input_size[2]), dtype="float32",
                               name="downsampled_input2")
    conv3 = tf.keras.layers.Conv2D(128, 3, activation="tanh", padding="same", name="conv3_block2")(X2)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation="tanh", padding="same", name="conv4_block2")(conv3)
    concat_block2 = tf.keras.layers.Concatenate(axis=-1, name="block2_concat")([pool, conv4])

    # Block 3
    conv1 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="conv1_block3")(concat_block2)
    conv2 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="conv2_block3")(conv1)
    pool = tf.keras.layers.MaxPool2D((2, 2), name="pool_block3")(conv2)

    X3 = tf.keras.layers.Input(shape=(input_size[0] / 8, input_size[1] / 8, input_size[2]), dtype="float32",
                               name="downsampled_input3")
    conv3 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="conv3_block3")(X3)
    conv4 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="conv4_block3")(conv3)
    concat_block3 = tf.keras.layers.Concatenate(axis=-1, name="block3_concat")([pool, conv4])

    # Block 4
    conv1 = tf.keras.layers.Conv2D(512, 3, activation="tanh", padding="same", name="conv1_block4")(concat_block3)
    conv2 = tf.keras.layers.Conv2D(512, 3, activation="tanh", padding="same", name="conv2_block4")(conv1)
    pool = tf.keras.layers.MaxPool2D((2, 2), name="pool_block4")(conv2)

    X4 = tf.keras.layers.Input(shape=(input_size[0] / 16, input_size[1] / 16, input_size[2]), dtype="float32",
                               name="downsampled_input4")
    conv3 = tf.keras.layers.Conv2D(512, 3, activation="tanh", padding="same", name="conv3_block4")(X4)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation="tanh", padding="same", name="conv4_block4")(conv3)
    concat_block4 = tf.keras.layers.Concatenate(axis=-1, name="block4_concat")([pool, conv4])

    # Group 2
    # Block 5
    conv1 = tf.keras.layers.Conv2D(2048, 3, activation="tanh",name="conv1_block5")(concat_block4)
    conv_block5 = tf.keras.layers.Conv2D(2048, 3, activation="tanh",name="block5_conv")(conv1)

    # UpSampling path
    # Group 3
    # Block 6
    deconv1 = tf.keras.layers.Conv2DTranspose(2048, 3, activation="tanh", name="deconv1_block6")(conv_block5)
    up1 = tf.keras.layers.Conv2D(2048, 3, activation="tanh", padding="same", name="up1_block6")(deconv1)
    up1 = tf.keras.layers.Conv2D(2048, 3, activation="tanh", padding="same", name="up2_block6")(up1)
    deconv2 = tf.keras.layers.Conv2DTranspose(1024, 3, activation="tanh", name="deconv2_block6")(up1)
    concat_block = tf.keras.layers.Concatenate(axis=-1, name="concat_block6")([deconv2, concat_block4])
    conv_block6 = tf.keras.layers.Conv2D(1024, 3, activation="tanh", padding="same", name="block6_conv")(concat_block)

    # Block 7
    up1 = tf.keras.layers.Conv2D(1028, 3, activation="tanh", padding="same", name="up1_block7")(conv_block6)
    up1 = tf.keras.layers.Conv2D(1028, 3, activation="tanh", padding="same", name="up2_block7")(up1)
    deconv2 = tf.keras.layers.Conv2DTranspose(512, 2, (2, 2), activation="tanh", name="deconv2_block7")(up1)
    concat_block = tf.keras.layers.Concatenate(axis=-1, name="concat_block7")([deconv2, concat_block3])
    conv_block7 = tf.keras.layers.Conv2D(512, 3, activation="tanh", padding="same", name="block7_conv")(concat_block)

    # Block 8
    up1 = tf.keras.layers.Conv2D(512, 3, activation="tanh", padding="same", name="up1_block8")(conv_block7)
    up1 = tf.keras.layers.Conv2D(512, 3, activation="tanh", padding="same", name="up2_block8")(up1)
    deconv2 = tf.keras.layers.Conv2DTranspose(256, 2, (2, 2), activation="tanh", name="deconv2_block8")(up1)
    concat_block = tf.keras.layers.Concatenate(axis=-1, name="concat_block8")([deconv2, concat_block2])
    conv_block8 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="block8_conv")(concat_block)

    # Block 9
    up1 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="up1_block9")(conv_block8)
    up1 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="up2_block9")(up1)
    deconv2 = tf.keras.layers.Conv2DTranspose(128, 2, (2, 2), activation="tanh", name="deconv2_block9")(up1)
    concat_block = tf.keras.layers.Concatenate(axis=-1, name="concat_block9")([deconv2, concat_block1])
    conv_block9 = tf.keras.layers.Conv2D(128, 3, activation="tanh", padding="same", name="block9_conv")(concat_block)

    # MULTI-OUTPUT PATHS
    # Group 4
    # Block 10 -> output of block 9
    deconvB10 = tf.keras.layers.Conv2DTranspose(64, 2, (2, 2), activation="tanh", name="deconv_block10")(conv_block9)
    convB10 = tf.keras.layers.Conv2D(64, 3, activation="tanh", padding="same", name="conv_block10")(deconvB10)

    # Block 11 -> output of block 8
    deconvB11 = tf.keras.layers.Conv2DTranspose(128, 4, (4, 4), activation="tanh", name="deconv_block11")(conv_block8)
    convB11 = tf.keras.layers.Conv2D(128, 3, activation="tanh", padding="same", name="conv_block11")(deconvB11)

    # Block 12 -> output of block 7
    deconvB12 = tf.keras.layers.Conv2DTranspose(256, 8, (8, 8), activation="tanh", name="deconv_block12")(conv_block7)
    convB12 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="conv_block12")(deconvB12)

    # Output Layers
    concat_out = tf.keras.layers.Concatenate(axis=-1, name="output_concat")([convB10, convB11, convB12])
    drop_out = tf.keras.layers.Dropout(0.5, name="drop_out")(concat_out)
    y = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="output_layer")(drop_out)
    return tf.keras.Model(inputs = [X, X1, X2, X3, X4], outputs = y)


def __MIMO_Net_ORIGINAL(input_size = (256, 256, 3)):
    """
    Function to implement the MIMO-Net and provide the tf.keras.Model
    for compilation and training/testing

    :param input_size: The size of the input image
    :return: tf.keras.Model of the MIMO-Net
    """
    X = tf.keras.layers.Input(shape=input_size, dtype = "float32", name = "main_input")

    # DownSampling path
    # Group 1:
    # Block 1
    conv1 = tf.keras.layers.Conv2D(64, 3, activation="tanh", padding="same", name="conv1_block1")(X)
    conv2 = tf.keras.layers.Conv2D(64, 3, activation="tanh", padding="same", name="conv2_block1")(conv1)
    pool = tf.keras.layers.MaxPool2D((2, 2), padding="same", name="pool_block1")(conv2)

    X1 = tf.keras.layers.Input(shape=(input_size[0] / 2, input_size[1] / 2, input_size[2]), dtype="float32",
                               name="downsampled_input1")
    conv3 = tf.keras.layers.Conv2D(64, 3, activation="tanh", padding="same", name="conv3_block1")(X1)
    conv4 = tf.keras.layers.Conv2D(64, 3, activation="tanh", padding="same", name="conv4_block1")(conv3)
    concat_block1 = tf.keras.layers.Concatenate(axis=-1, name="block1_concat")([pool, conv4])

    # Block 2
    conv1 = tf.keras.layers.Conv2D(128, 3, activation="tanh", padding="same", name="conv1_block2")(concat_block1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation="tanh", padding="same", name="conv2_block2")(conv1)
    pool = tf.keras.layers.MaxPool2D((2, 2), padding="same", name="pool_block2")(conv2)

    X2 = tf.keras.layers.Input(shape=(input_size[0] / 4, input_size[1] / 4, input_size[2]), dtype="float32",
                               name="downsampled_input2")
    conv3 = tf.keras.layers.Conv2D(128, 3, activation="tanh", padding="same", name="conv3_block2")(X2)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation="tanh", padding="same", name="conv4_block2")(conv3)
    concat_block2 = tf.keras.layers.Concatenate(axis=-1, name="block2_concat")([pool, conv4])

    # Block 3
    conv1 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="conv1_block3")(concat_block2)
    conv2 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="conv2_block3")(conv1)
    pool = tf.keras.layers.MaxPool2D((2, 2), padding="same", name="pool_block3")(conv2)

    X3 = tf.keras.layers.Input(shape=(input_size[0] / 8, input_size[1] / 8, input_size[2]), dtype="float32",
                               name="downsampled_input3")
    conv3 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="conv3_block3")(X3)
    conv4 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="conv4_block3")(conv3)
    concat_block3 = tf.keras.layers.Concatenate(axis=-1, name="block3_concat")([pool, conv4])

    # Block 4
    conv1 = tf.keras.layers.Conv2D(512, 3, activation="tanh", padding="same", name="conv1_block4")(concat_block3)
    conv2 = tf.keras.layers.Conv2D(512, 3, activation="tanh", padding="same", name="conv2_block4")(conv1)
    pool = tf.keras.layers.MaxPool2D((2, 2), padding="same", name="pool_block4")(conv2)

    X4 = tf.keras.layers.Input(shape=(input_size[0] / 16, input_size[1] / 16, input_size[2]), dtype="float32",
                               name="downsampled_input4")
    conv3 = tf.keras.layers.Conv2D(512, 3, activation="tanh", padding="same", name="conv3_block4")(X4)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation="tanh", padding="same", name="conv4_block4")(conv3)
    concat_block4 = tf.keras.layers.Concatenate(axis=-1, name="block4_concat")([pool, conv4])

    # Group 2
    # Block 5
    conv1 = tf.keras.layers.Conv2D(2048, 3, activation="tanh",name="conv1_block5")(concat_block4)
    conv_block5 = tf.keras.layers.Conv2D(2048, 3, activation="tanh",name="block5_conv")(conv1)

    # UpSampling path
    # Group 3
    # Block 6
    deconv1 = tf.keras.layers.Conv2DTranspose(2048, 3, activation="tanh", name="deconv1_block6")(conv_block5)
    up1 = tf.keras.layers.Conv2D(2048, 3, activation="tanh", padding="same", name="up1_block6")(deconv1)
    up1 = tf.keras.layers.Conv2D(2048, 3, activation="tanh", padding="same", name="up2_block6")(up1)
    deconv2 = tf.keras.layers.Conv2DTranspose(1024, 3, activation="tanh", name="deconv2_block6")(up1)
    up_skip = tf.keras.layers.Conv2DTranspose(1024, 2, (2, 2), activation="tanh", name="up_skip_block6")(concat_block4)
    concat_block = tf.keras.layers.Concatenate(axis=-1, name="concat_block6")([deconv2, up_skip])
    conv_block6 = tf.keras.layers.Conv2D(1024, 3, activation="tanh", padding="same", name="block6_conv")(concat_block)

    # Block 7
    deconv1 = tf.keras.layers.Conv2DTranspose(1024, 3, activation="tanh", name="deconv1_block7")(conv_block6)
    up1 = tf.keras.layers.Conv2D(1028, 3, activation="tanh", padding="same", name="up1_block7")(deconv1)
    up1 = tf.keras.layers.Conv2D(1028, 3, activation="tanh", padding="same", name="up2_block7")(up1)
    deconv2 = tf.keras.layers.Conv2DTranspose(512, 3, activation="tanh", name="deconv2_block7")(up1)
    up_skip = tf.keras.layers.Conv2DTranspose(512, 2, (2, 2), activation="tanh", name="up_skip_block7")(concat_block3)
    concat_block = tf.keras.layers.Concatenate(axis=-1, name="concat_block7")([deconv2, up_skip])
    conv_block7 = tf.keras.layers.Conv2D(512, 3, activation="tanh", padding="same", name="block7_conv")(concat_block)

    # Block 8
    deconv1 = tf.keras.layers.Conv2DTranspose(512, 3, activation="tanh", name="deconv1_block8")(conv_block7)
    up1 = tf.keras.layers.Conv2D(512, 3, activation="tanh", padding="same", name="up1_block8")(deconv1)
    up1 = tf.keras.layers.Conv2D(512, 3, activation="tanh", padding="same", name="up2_block8")(up1)
    deconv2 = tf.keras.layers.Conv2DTranspose(256, 3, activation="tanh", name="deconv2_block8")(up1)
    up_skip = tf.keras.layers.Conv2DTranspose(256, 2, (2, 2), activation="tanh", name="up_skip_block8")(concat_block2)
    concat_block = tf.keras.layers.Concatenate(axis=-1, name="concat_block8")([deconv2, up_skip])
    conv_block8 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="block8_conv")(concat_block)

    # Block 9
    deconv1 = tf.keras.layers.Conv2DTranspose(256, 3, activation="tanh", name="deconv1_block9")(conv_block8)
    up1 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="up1_block9")(deconv1)
    up1 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="up2_block9")(up1)
    deconv2 = tf.keras.layers.Conv2DTranspose(128, 3, activation="tanh", name="deconv2_block9")(up1)
    up_skip = tf.keras.layers.Conv2DTranspose(128, 2, (2, 2), activation="tanh", name="up_skip_block9")(concat_block1)
    concat_block = tf.keras.layers.Concatenate(axis=-1, name="concat_block9")([deconv2, up_skip])
    conv_block9 = tf.keras.layers.Conv2D(128, 3, activation="tanh", padding="same", name="block9_conv")(concat_block)

    # MULTI-OUTPUT PATHS
    # Group 4
    # Block 10 -> output of block 9
    deconvB10 = tf.keras.layers.Conv2DTranspose(64, 2, (2, 2), activation="tanh", name="deconv_block10")(conv_block9)
    convB10 = tf.keras.layers.Conv2D(64, 3, activation="tanh", padding="same", name="conv_block10")(deconvB10)

    # Block 11 -> output of block 8
    deconvB11 = tf.keras.layers.Conv2DTranspose(128, 4, (4, 4), activation="tanh", name="deconv_block11")(conv_block8)
    convB11 = tf.keras.layers.Conv2D(128, 3, activation="tanh", padding="same", name="conv_block11")(deconvB11)

    # Block 12 -> output of block 7
    deconvB12 = tf.keras.layers.Conv2DTranspose(256, 8, (8, 8), activation="tanh", name="deconv_block12")(conv_block7)
    convB12 = tf.keras.layers.Conv2D(256, 3, activation="tanh", padding="same", name="conv_block12")(deconvB12)

    # Output Layers
    concat_out = tf.keras.layers.Concatenate(axis=-1, name="output_concat")([convB10, convB11, convB12])
    drop_out = tf.keras.layers.Dropout(0.5, name="drop_out")(concat_out)
    y = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="output_layer")(drop_out)
    y = tf.squeeze(y)
    return tf.keras.Model(inputs = [X, X1, X2, X3, X4], outputs = y)







