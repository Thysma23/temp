from typing import Literal
import cv2
import numpy as np
import tensorflow as tf
from config import DATASET_PATH, RESIZE_DIM
from sklearn.model_selection import train_test_split
from keras.api.utils import to_categorical


def load_dataset(category: Literal["train", "test"] = "train", flatten: bool = False,  print_repartition: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Load images from NORMAL, BACTERIA, and VIRUS classes.

    Args:
        category (str): The category of images to load (train or test).
        flatten (bool): Whether to flatten the images or not.
        print_repartition (bool): Whether to print the class distribution.

    Returns:
        tuple (ndarray, ndarray): A tuple containing the images and their corresponding labels.
    """
    category_path = DATASET_PATH / category
    normal_path = category_path / "NORMAL"
    pneumonia_path = category_path / "PNEUMONIA"

    images = []
    labels = []

    label_map = {"NORMAL": 0, "BACTERIA": 1, "VIRUS": 2}

    # Load NORMAL images
    for img_path in normal_path.glob("*"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:  # Check if image was loaded correctly
            print(f"Warning: Could not read image {img_path}")
            continue

        img = cv2.resize(img, RESIZE_DIM)
        img = img.astype("float32") / 255.0  # Normalize
        images.append(img.flatten() if flatten else img)
        labels.append(label_map["NORMAL"])

    # Load PNEUMONIA images (distinguish BACTERIA & VIRUS)
    for img_path in pneumonia_path.glob("*"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        img = cv2.resize(img, RESIZE_DIM)
        img = img.astype("float32") / 255.0

        if "bacteria" in img_path.name.lower():
            labels.append(label_map["BACTERIA"])
        elif "virus" in img_path.name.lower():
            labels.append(label_map["VIRUS"])

        images.append(img.flatten() if flatten else img)

    if print_repartition:
        unique, counts = np.unique(labels, return_counts=True)
        unique_labels = list(label_map.keys())
        class_distribution = {unique_labels[i]: int(
            counts[i]) for i in range(len(unique))}
        print(f"Class distribution in {category}: {class_distribution}")

    return np.array(images), np.array(labels)


def split_dataset(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into training and testing sets.
    """
    return train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)  # type: ignore


def resize_images(**x: np.ndarray) -> list[np.ndarray]:
    """
    Resize images to the specified dimensions.
    Then normalize the images to have pixel values between 0 and 1 to improve performance (images are in grayscale).
    """
    resized_images = []
    for (key, images) in x.items():
        # Print dataset shapes before resizing
        print(f"BEFORE x_{key}: {images.shape}")

        # Resize images
        images = np.array([cv2.resize(img, RESIZE_DIM) for img in images])

        # Normalize images for better performance
        images = images.reshape(
            images.shape[0], RESIZE_DIM[0], RESIZE_DIM[1], 1)

        # Print dataset shapes after resizing
        print(f"THEN x_{key}: {images.shape}")
        resized_images.append(images)
    return resized_images


def categorize_labels(**y: np.ndarray) -> list[np.ndarray]:
    """
    Resize labels to categorical format.
    """
    resized_labels = []
    for (key, labels) in y.items():
        # Print dataset shapes before resizing
        labels = to_categorical(labels, num_classes=3)
        # Print dataset shapes after resizing
        print(f"y_{key} shape: {labels.shape}")
        resized_labels.append(labels)
    return resized_labels
