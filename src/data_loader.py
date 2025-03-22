import pandas as pd
import yaml

def load_config(config_path):
    """
    Load configuration
    """
    with open(config_path,"r") as f:
        config = yaml.safe_load(f)
    return config

def load_data(file_path):
    """
    Load Dataset from path mentioned in the config.yaml file
    """
    return pd.read_csv(file_path)