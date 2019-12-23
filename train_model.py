# import necessary packages
import argparse

import yaml

import train_utilities

DESCRIPTION = """
This is a train API
"""
VERSION = "0.01alpha"

if __name__ == "__main__":

    # For help
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    # Add options
    parser.add_argument("-V", "--version", help="Shows program version", action="store_true")
    parser.add_argument("-L", "--load-model", help="The model h5 file to load")
    parser.add_argument("-R", "--resume-epoch", help="The epoch to resume train")
    parser.add_argument("-W", "--loading-weights",
                        help="Status telling if we are loading weights or not. Give 1 if loading weights, else ignore it")
    parser.add_argument("-C", "--config-file", help="The YAML config file for training")
    # Read args
    args = parser.parse_args()

    # check for version
    if args.version:
        print("Using Version %s" % (VERSION))
        exit(1)

    load_model_path = ""
    resume_epoch = 0
    config_file_path = "train_config.yaml"
    loading_weights = False

    if args.load_model:
        load_model_path = str(args.load_model)
    if args.resume_epoch:
        resume_epoch = int(args.resume_epoch)
    if args.config_file:
        config_file_path = str(args.config_file)
    if args.loading_weights:
        if int(args.loading_weights) == 1:
            loading_weights = True

    print("[INFO]: Parsing Config file")
    config = yaml.load(open(config_file_path, mode="r"))

    print("[INFO]: The following configuration is parsed")
    for k, v in config.items():
        print("{0}: {1}".format(k, v))

    choice = int(input("[PROBE]: Do you wish to proceed (1/0): "))
    print(choice)
    if choice != 1:
        print("[INFO]: Aborting train operations")
        exit(1)

    if load_model_path == "":
        train_utilities.new_train_model(config=config)
        print("[INFO]: Proceeding to resume training from {0} epoch".format(resume_epoch))
    else:
        print("[INFO]: Proceeding to train {0} model from scratch".format(config["model_name"]))
        train_utilities.resume_train_model(model_path=load_model_path, resume_epoch=resume_epoch, config=config)
