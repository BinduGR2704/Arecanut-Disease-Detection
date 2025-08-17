import os
import yaml
from cnnClassifier import logging
import json
import joblib
from ensure import ensure_annotations
# from typeguard import typechecked as ensure_annotations
from box import Box
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> Box:
    """Reads a YAML file and returns its contents as a Box.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        Exception: If any other error occurs while reading the YAML file.

    Returns:
        Box: Box type containing the parsed YAML content.
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError("YAML file is empty")
            logging.info(f"YAML file: {path_to_yaml} loaded successfully")
            return Box(content)
    except Exception as e:
        logging.error(f"Error reading YAML file {path_to_yaml}: {str(e)}")
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories if they don't already exist.

    Args:
        path_to_directories (list): List of directory paths to be created.
        verbose (bool, optional): Whether to log the creation of each directory. Defaults to True.
    """
    for path in path_to_directories:
        try:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logging.info(f"Created directory at: {path}")
        except Exception as e:
            logging.error(f"Failed to create directory at {path}: {str(e)}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves a dictionary as a JSON file.

    Args:
        path (Path): Path to the JSON file.
        data (dict): Data to be saved in the JSON file.
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logging.info(f"JSON file saved at: {path}")
    except Exception as e:
        logging.error(f"Failed to save JSON file at {path}: {str(e)}")


@ensure_annotations
def load_json(path: Path) -> Box:
    """Loads JSON data from a file and returns it as a Box.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        Box: Data loaded from the JSON file.
    """
    try:
        with open(path, "r") as f:
            content = json.load(f)
        logging.info(f"JSON file loaded successfully from: {path}")
        return Box(content)
    except Exception as e:
        logging.error(f"Failed to load JSON file from {path}: {str(e)}")
        raise e


@ensure_annotations
def save_bin(data: Any, path: Path):
    """Saves data as a binary file.

    Args:
        data (Any): Data to be saved as a binary file.
        path (Path): Path to save the binary file.
    """
    try:
        joblib.dump(value=data, filename=path)
        logging.info(f"Binary file saved at: {path}")
    except Exception as e:
        logging.error(f"Failed to save binary file at {path}: {str(e)}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads binary data from a file.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: Data loaded from the binary file.
    """
    try:
        data = joblib.load(path)
        logging.info(f"Binary file loaded from: {path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load binary file from {path}: {str(e)}")
        raise e


@ensure_annotations
def get_size(path: Path) -> str:
    """Returns the size of the file in KB.

    Args:
        path (Path): Path to the file.

    Returns:
        str: File size in KB.
    """
    try:
        size_in_kb = round(os.path.getsize(path) / 1024)
        return f"~ {size_in_kb} KB"
    except Exception as e:
        logging.error(f"Failed to get size of the file at {path}: {str(e)}")
        raise e


def decode_image(imgstring: str, file_name: str):
    """Decodes a base64 string and saves it as an image file.

    Args:
        imgstring (str): Base64-encoded image string.
        file_name (str): Path to save the decoded image.
    """
    try:
        img_data = base64.b64decode(imgstring)
        with open(file_name, 'wb') as f:
            f.write(img_data)
        logging.info(f"Image saved as {file_name}")
    except Exception as e:
        logging.error(f"Failed to decode and save image {file_name}: {str(e)}")
        raise e


def encode_image_into_base64(cropped_image_path: str) -> str:
    """Encodes an image into a base64 string.

    Args:
        cropped_image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded image string.
    """
    try:
        with open(cropped_image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        logging.info(f"Image encoded to base64 from {cropped_image_path}")
        return encoded_image
    except Exception as e:
        logging.error(f"Failed to encode image to base64 from {cropped_image_path}: {str(e)}")
        raise e
