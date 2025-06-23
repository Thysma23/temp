"""
    This module provides functions to load and preprocess the Chest X-ray dataset for TensorFlow models (3 dimensions). (TFDataset).
"""
import numpy as np
import tensorflow as tf
from src.datasets.datasetHelper import load_dataset, split_dataset, resize_images, categorize_labels
from config import IS_DATA_AUGMENTED


def load_data_with_validation():
    """
    Load and preprocess the dataset, splitting it into training, validation, and test sets.

    Returns:
        tuple: ((x_train, y_train), (x_val, y_val), (x_test, y_test))
                Preprocessed training, validation, and test data with their labels.
    """
    # Load training data
    print("Loading training data...")
    x_train_raw, y_train_raw = load_dataset('train', print_repartition=True)

    # Load test data
    print("Loading test data...")
    x_test_raw, y_test_raw = load_dataset('test', print_repartition=True)

    # Split training data to create validation set
    print("Creating validation set...")
    x_train, x_val, y_train, y_val = split_dataset(x_train_raw, y_train_raw)

    # Resize and normalize images
    print("Preprocessing images...")
    x_train, x_val, x_test = resize_images(
        train=x_train, val=x_val, test=x_test_raw)

    # Convert labels to categorical (one-hot encoding)
    print("Processing labels...")
    y_train, y_val, y_test = categorize_labels(
        train=y_train, val=y_val, test=y_test_raw)

    print("Data loading complete!")

    if IS_DATA_AUGMENTED:
        print("Applying data augmentation...")
        x_train = np.append(x_train, __augment_data(x_train), axis=0)
        y_train = np.append(y_train, y_train, axis=0)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_data():
    """
    Load and preprocess the dataset, splitting it into training, and test sets.

    Returns:
        tuple: ((x_train, y_train), (x_test, y_test))
                Preprocessed training and test data with their labels.
    """
    # Load training data
    print("Loading training data...")
    x_train_raw, y_train_raw = load_dataset('train', print_repartition=True)

    # Load test data
    print("Loading test data...")
    x_test_raw, y_test_raw = load_dataset('test', print_repartition=True)

    # Resize and normalize images
    print("Preprocessing images...")
    x_train, x_test = resize_images(train=x_train_raw, test=x_test_raw)

    # Convert labels to categorical (one-hot encoding)
    print("Processing labels...")
    y_train, y_test = categorize_labels(train=y_train_raw, test=y_test_raw)

    print("Data loading complete!")

    if IS_DATA_AUGMENTED:
        print("Applying data augmentation...")
        x_train = np.append(x_train, __augment_data(x_train), axis=0)
        y_train = np.append(y_train, y_train, axis=0)

    return (x_train, y_train), (x_test, y_test)

def __augment_data(x_train: np.ndarray) -> np.ndarray:
    """ Apply data augmentation using a loop.

    Args:
        x_train (np.ndarray): Training data.

    Returns:
        np.ndarray: Augmented training data.
    """
    augmented_images = []
    for image in x_train:
        augmented_image = __data_augmentation(image)
        augmented_images.append(augmented_image)

    return np.array(augmented_images)


def __data_augmentation(image: np.ndarray) -> np.ndarray:
    """ Apply data augmentation transformations.

    Args:
        image (tf.Tensor): Input image tensor.

    Returns:
        tf.Tensor: Augmented image tensor.
    """
    image = tf.image.random_flip_left_right(image)  # type: ignore
    image = tf.image.random_brightness(
        image, max_delta=0.2)  # type: ignore
    image = tf.image.random_contrast(
        image, lower=0.8, upper=1.2)  # type: ignore
    return image