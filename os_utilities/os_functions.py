# Import necessary functions
import os
import traceback


def make_directories(paths: list):
    """
    Function to make directories mentioned in the
    list of paths and all its subdirectories

    :param paths: A list of paths to create

    :return: None
    """
    for path in paths:
        try:
            os.makedirs(path)
        except OSError:
            print("[WARN]: Directory: %s already exist" % (path))
        except Exception as err:
            print("[ERROR]: ", err)
            print("[ERROR]: ", traceback.print_exc())
