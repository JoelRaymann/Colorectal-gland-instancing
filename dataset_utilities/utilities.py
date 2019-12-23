import os

# NOTE: Private APIs
def __validate_crag_dataset(dataset_path: str) -> bool:
    """
    Function to validate the CRAG dataset for all its components
    
    Arguments:
        dataset_path {str} -- The root path of the dataset
    
    Returns:
        bool -- the result
    """
    if all(element in os.listdir(dataset_path) for element in ['valid', 'train']):
        check_dir = ['Images', 'Annotation', 'Overlay']
        if all(element in os.listdir(dataset_path + "/train") and element in os.listdir(dataset_path + "/valid") for element in check_dir):
            return True
        else:
            return False
    else:
        return False

def generate_dataset_path(dataset_path: str, dataset_family: str) -> tuple:
    """
    Function to generate the list of paths for all the train and valid set for 
    the given set of family
    
    Arguments:
        dataset_path {str} -- the root path of the dataset
        dataset_family {str} -- The family of the dataset - CRAG or GLaS
    
    Returns:
        tuple -- (train_set, valid_set)
    """
    # Check family

    # CRAG Family 
    if dataset_family == "CRAG":

        print("[INFO]: Using CRAG dataset. Version: 2.0")
        if __validate_crag_dataset(dataset_path):
            print("[INFO]: Dataset Verified")
        else:
            print("[ERROR]: Invalid dataset, Check the dataset path")
            raise FileNotFoundError
    
        print("[INFO]: Generating Training Path list")
        
        train_images = [dataset_path + "/train/Images/" + path for path in os.listdir(dataset_path + "/train/Images") if path[0] != "."]
        train_set = [(path, path.replace("Images", "Annotation"), path.replace("Images", "Overlay")) for path in train_images]
        
        print("[INFO]: Generating Valid Path list")

        valid_images = [dataset_path + "/valid/Images/" + path for path in os.listdir(dataset_path + "/valid/Images") if path[0] != "."]
        valid_set = [(path, path.replace("Images", "Annotation"), path.replace("Images", "Overlay")) for path in valid_images]
        return train_set, valid_set
    
    # TODO: // Setup GLaS Dataset
    else:
        print("[TODO]: YET TO SET GLaS Dataset")
        return None


