"""
    This module provides functions to load and preprocess the Chest X-ray dataset for Scikit-learn models (1 dimension (flattened)) (SKDataset).
"""
import numpy as np
from src.datasets.datasetHelper import load_dataset, split_dataset, resize_images, categorize_labels
from config import IS_DATA_AUGMENTED
from config import RESIZE_DIM

def load_data_with_validation():
    """
    Load and preprocess the dataset, splitting it into training, validation, and test sets.

    Returns:
        tuple: ((x_train, y_train), (x_val, y_val), (x_test, y_test))
                Preprocessed training, validation, and test data with their labels.
    """
    # Load training data
    print("Loading training data...")
    x_train_raw, y_train_raw = load_dataset(
        'train', flatten=True, print_repartition=True)

    # Load test data
    print("Loading test data...")
    x_test_raw, y_test_raw = load_dataset(
        'test', flatten=True, print_repartition=True)

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
        x_train = np.append(x_train, augment_data(x_train), axis=0)
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
    x_train, y_train = load_dataset(
        'train', flatten=True, print_repartition=True)

    # Load test data
    print("Loading test data...")
    x_test, y_test = load_dataset(
        'test', flatten=True, print_repartition=True)

    print("Data loading complete!")

    if IS_DATA_AUGMENTED:
        print("Applying data augmentation...")
        x_train = np.append(x_train, augment_data(x_train), axis=0)
        y_train = np.append(y_train, y_train, axis=0)

    return (x_train, y_train), (x_test, y_test)

def _infer_channels(flat_length: int, h: int, w: int) -> int:
    """
    Devine le nombre de canaux à partir de la taille du vecteur aplati.
    l = h * w         ➜  1 canal
    l = h * w * 3     ➜  3 canaux
    Sinon lève ValueError.
    """
    if flat_length == h * w:
        return 1
    if flat_length == h * w * 3:
        return 3
    raise ValueError(
        f"Impossible de déduire le nombre de canaux : "
        f"len={flat_length} / (h,w)=({h},{w})"
    )

def augment_data(x_train: np.ndarray) -> np.ndarray:
    """
    Génère un jeu de données augmenté à partir des images déjà aplaties
    (n_samples, n_features).

    Retourne un tableau (n_samples, n_features) prêt pour Scikit-learn.
    """
    return np.stack([__data_augmentation(img) for img in x_train], axis=0)


def __data_augmentation(image: np.ndarray) -> np.ndarray:
    """
    Applique flip horizontal, luminosité et contraste puis ré-aplatit.

    Args:
        image (np.ndarray): vecteur 1-D (flatten) représentant une image
                            de taille RESIZE_DIM.

    Returns:
        np.ndarray: vecteur 1-D transformé (même longueur).
    """
    h, w = RESIZE_DIM
    c     = _infer_channels(len(image), h, w)

    orig_dtype = image.dtype
    img = image.astype(np.float32)
    if orig_dtype == np.uint8:
        img /= 255.0

    # Remise en forme : (H, W, C) si C>1, sinon (H, W)
    img = img.reshape((h, w, c)) if c > 1 else img.reshape((h, w))

    # ---------------------------- 1. Flip horizontal ----------------------------
    if np.random.rand() < 0.5:
        img = np.flip(img, axis=1)  # ↔

    # ---------------------------- 2. Luminosité ---------------------------------
    delta = np.random.uniform(-0.20, 0.20)
    img = np.clip(img + delta, 0.0, 1.0)

    # ---------------------------- 3. Contraste ----------------------------------
    factor = np.random.uniform(0.80, 1.20)
    mean = img.mean()
    img = np.clip((img - mean) * factor + mean, 0.0, 1.0)

    # --------------------------------------------------------------------------- #
    # Ré-aplatissage + retour au type initial                                     #
    # --------------------------------------------------------------------------- #
    img = img.reshape(-1)
    if orig_dtype == np.uint8:
        img = (img * 255.0).astype(np.uint8)

    return img