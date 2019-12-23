
# Import the necessary packages
import sys
import traceback

import tensorflow as tf

import dataset_utilities
import metrics
# Import in-house packages
import models
import os_utilities


def new_train_model(config: dict):
    """
    Function to train a new model from scratch

    :param config: A dictionary with the configuration to train
    {
        "no_of_epochs" : no of epochs to train
        "learning_rate" : The learning rate
        "steps_per_epoch": The total steps per epoch
        "train_batch_size" : The batch size to use for training
        "test_batch_size" : The batch size to use for testing
        "threads" : The no. of threads to use
        "GPUs" : The total number of GPUs to use
        "dataset_path": The path of the dataset
        "dataset_family": The type of dataset used
        "model_name" : The model name, NOTE: This will be used to save the model.
    }

    :return: None
    """

    # Get required configuration to train
    no_of_epochs = config["no_of_epochs"]
    learning_rate = config["learning_rate"]
    train_batch_size = config["train_batch_size"]
    test_batch_size = config["test_batch_size"]
    model_name = config["model_name"]
    GPUs = config["GPUs"]
    threads = config["threads"]
    dataset_path = config["dataset_path"]
    dataset_family = config["dataset_family"]
    steps_per_epoch = config["steps_per_epoch"]
    image_width = config["image_width"]
    image_height = config["image_height"]
    resize = config["resize"]

    print("[INFO]: Using config: \n", config)

    # Set up environments
    folders = [
        "./model_save/{0}/checkpoints/".format(model_name),
        "./output_logs/{0}/csv_log/".format(model_name),
        "./model_save/{0}/best_model/".format(model_name),
        "./model_save/{0}/saved_model/".format(model_name),
        "./output_logs/{0}/graphs/".format(model_name)
    ]
    print("[INFO]: Setting up folders: ")
    os_utilities.make_directories(paths = folders)

    # Load dataset
    # Get path set
    print("[INFO]: Reading Dataset Paths")
    train_set, test_set = dataset_utilities.generate_dataset_path(dataset_path=dataset_path, dataset_family=dataset_family)

    print("[INFO]: Setting up train and test generators")
    train_gen = dataset_utilities.DataGenerator(train_set,
                                                batch_size=train_batch_size,
                                                image_size=(image_width, image_height),
                                                resize = resize,
                                                model_name = model_name,
                                                aug_config_path="./augmentation_config.yaml",
                                                apply_augmentation=True, shuffle=True)

    test_gen = dataset_utilities.DataGenerator(test_set,
                                               batch_size=test_batch_size,
                                               image_size=(image_width, image_height),
                                               resize = resize,
                                               model_name=model_name,
                                               aug_config_path="./augmentation_config.yaml",
                                               apply_augmentation=False, shuffle=False)

    # Get the model
    model = models.ModelSelector(model_name = model_name, input_size = (image_width, image_height, 3))

    # Handle multi-GPU architecture
    if GPUs > 1:
        strategy = tf.distribute.MirroredStrategy()
        print("[INFO]: No. of GPUs used for training: ", strategy.num_replicas_in_sync)

        with strategy.scope():
            model.compile(optimizer =  tf.keras.optimizers.Adam(lr=learning_rate),
                          loss = "binary_crossentropy",
                          metrics = ["accuracy", metrics.dice_coef])

    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                      loss="binary_crossentropy",
                      metrics=["accuracy", metrics.dice_coef])

    # Set-up Callbacks
    csv_callback = tf.keras.callbacks.CSVLogger("./output_logs/{0}/csv_log/{0}_log.csv".format(model_name))
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = "./{0}_logs".format(model_name))
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("./model_save/{0}/checkpoints/{0}_checkpoint.h5".format(model_name), period = 1, save_weights_only=True)
    best_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("./model_save/{0}/best_model/best_{0}_checkpoint.h5".format(model_name), save_best_only = True, save_weights_only=True)
    model_save_path = "./model_save/{0}/saved_model/".format(model_name)

    # Train model
    try:
        model.fit_generator(train_gen,
                            epochs=no_of_epochs,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=[csv_callback, checkpoint_callback, best_model_checkpoint_callback],
                            validation_data=test_gen,
                            workers=threads,
                            use_multiprocessing=True)

    except KeyboardInterrupt:
        print("\n[INFO] Train Interrupted")
        model.save_weights(model_save_path + "{0}_interrupted.h5".format(model_name))
        sys.exit(2)

    except Exception as err:
        print("\n{CRITICAL}: Error, UnHandled Exception: ", err, "\n", traceback.print_exc())
        print("{CRITICAL}: Trying to save the model")
        model.save_weights(model_save_path + "{0}_error.h5".format(model_name))
        del model
        sys.exit(2)

        # Model saving
    model.save_weights(filepath=model_save_path + "{0}_weights.h5".format(model_name))
    model.save(filepath=model_save_path + "{0}.h5".format(model_name))

    # Testing Results
    print("[+] Testing the model")
    loss, accuracy, dice_index = model.evaluate_generator(test_gen, verbose=1)
    print("[+] Test Loss: ", loss)
    print("[+] Test Accuracy: ", accuracy)
    print("[+] Dice Index: ", dice_index)

def resume_train_model(model_path: str, resume_epoch:int, config: dict):
    """
    Function to resume a training for the model

    :param model_path: The path to load the model weights
    :param resume_epoch: The epoch to resume training from
    :param config: A dictionary with the configuration to train
    {
        "no_of_epochs" : no of epochs to train
        "learning_rate" : The learning rate
        "steps_per_epoch": The total steps per epoch
        "train_batch_size" : The batch size to use for training
        "test_batch_size" : The batch size to use for testing
        "threads" : The no. of threads to use
        "GPUs" : The total number of GPUs to use
        "dataset_path": The path of the dataset
        "dataset_family": The type of dataset used
        "model_name" : The model name, NOTE: This will be used to save the model.
    }

    :return: None
    """

    # Get required configuration to train
    no_of_epochs = config["no_of_epochs"]
    learning_rate = config["learning_rate"]
    train_batch_size = config["train_batch_size"]
    test_batch_size = config["test_batch_size"]
    model_name = config["model_name"]
    GPUs = config["GPUs"]
    threads = config["threads"]
    dataset_path = config["dataset_path"]
    dataset_family = config["dataset_family"]
    steps_per_epoch = config["steps_per_epoch"]
    image_width = config["image_width"]
    image_height = config["image_height"]
    resize = config["resize"]

    print("[INFO]: Using config: \n", config)

    # Set up environments
    folders = [
        "./model_save/{0}/checkpoints/".format(model_name),
        "./output_logs/{0}/csv_log/".format(model_name),
        "./model_save/{0}/best_model/".format(model_name),
        "./model_save/{0}/saved_model/".format(model_name),
        "./output_logs/{0}/graphs/".format(model_name)
    ]
    print("[INFO]: Setting up folders: ")
    os_utilities.make_directories(paths = folders)

    # Load dataset
    # Get path set
    print("[INFO]: Reading Dataset Paths")
    train_set, test_set = dataset_utilities.generate_dataset_path(dataset_path=dataset_path, dataset_family=dataset_family)

    print("[INFO]: Setting up train and test generators")
    train_gen = dataset_utilities.DataGenerator(train_set,
                                                batch_size=train_batch_size,
                                                image_size=(image_width, image_height),
                                                resize = resize,
                                                model_name=model_name,
                                                aug_config_path="./augmentation_config.yaml",
                                                apply_augmentation=True, shuffle=True)

    test_gen = dataset_utilities.DataGenerator(test_set,
                                               batch_size=test_batch_size,
                                               image_size=(image_width, image_height),
                                               resize = resize,
                                               model_name=model_name,
                                               aug_config_path="./augmentation_config.yaml",
                                               apply_augmentation=False, shuffle=False)

    # Get the model
    model = models.ModelSelector(model_name = model_name, input_size = (image_width, image_height, 3))

    # Handle multi-GPU architecture
    if GPUs > 1:
        strategy = tf.distribute.MirroredStrategy()
        print("[INFO]: No. of GPUs used for training: ", strategy.num_replicas_in_sync)

        with strategy.scope():
            model.compile(optimizer =  tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss = "binary_crossentropy",
                          metrics = ["accuracy", metrics.dice_coef])
            model.load_weights(model_path)

    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="binary_crossentropy",
                      metrics=["accuracy", metrics.dice_coef])
        model.load_weights(model_path)

    # Set-up Callbacks
    csv_callback = tf.keras.callbacks.CSVLogger("./output_logs/{0}/csv_log/{0}_log.csv".format(model_name))
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = "./{0}_logs".format(model_name))
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("./model_save/{0}/checkpoints/{0}_checkpoint.h5".format(model_name), period = 1, save_weights_only=True)
    best_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("./model_save/{0}/best_model/best_{0}_checkpoint.h5".format(model_name), save_best_only = True, save_weights_only=True)
    model_save_path = "./model_save/{0}/saved_model/".format(model_name)

    # Train model
    try:
        model.fit_generator(train_gen,
                            epochs=no_of_epochs,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=[csv_callback, checkpoint_callback, best_model_checkpoint_callback],
                            validation_data=test_gen,
                            workers=threads,
                            initial_epoch=resume_epoch,
                            use_multiprocessing=True)

    except KeyboardInterrupt:
        print("\n[INFO] Train Interrupted")
        model.save_weights(model_save_path + "{0}_interrupted.h5".format(model_name))
        sys.exit(2)

    except Exception as err:
        print("\n{CRITICAL}: Error, UnHandled Exception: ", err, "\n", traceback.print_exc())
        print("{CRITICAL}: Trying to save the model")
        model.save_weights(model_save_path + "{0}_error.h5".format(model_name))
        del model
        sys.exit(2)

        # Model saving
    model.save_weights(filepath=model_save_path + "{0}_weights.h5".format(model_name))
    model.save(filepath=model_save_path + "{0}.h5".format(model_name))

    # Testing Results
    print("[+] Testing the model")
    loss, accuracy, dice_index = model.evaluate_generator(test_gen, verbose=1)
    print("[+] Test Loss: ", loss)
    print("[+] Test Accuracy: ", accuracy)
    print("[+] Dice Index: ", dice_index)