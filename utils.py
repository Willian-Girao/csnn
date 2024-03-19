import os, datetime

def create_directory(directory_path):
    """
    Create a directory if it does not exist already.
    
    Args:
    - directory_path (str): The path of the directory to be created.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def create_results_dir():
    create_directory('trained_models')
    
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = os.path.join('trained_models', datetime_str)
    
    create_directory(model_path)

    return model_path
