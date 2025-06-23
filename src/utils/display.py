from pathlib import Path
from typing import Dict, List, Literal
from matplotlib import pyplot as plt
import cv2
import numpy as np
import seaborn as sns
from config import DATASET_PATH
from sklearn.metrics import confusion_matrix
from config import RESIZE_DIM


def display_images(
        img_paths: List[Path],
        title: str,
        output_length: int = 5,
):
    """
    Display images in a row format.
    - `img_paths`: List of image paths to display
    - `title`: Title for the images
    """
    _fig, axes = plt.subplots(1, output_length, figsize=(15, 5))
    for i, img_path in enumerate(img_paths[:output_length]):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, RESIZE_DIM)
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(title)
    plt.show()


def show_sample_images(num_images: int, category: Literal["train", "test"] = "train"):
    """
    Show example images from NORMAL, BACTERIA, and VIRUS classes.
    - `category`: train, test
    - `num_images`: number of images to display
    """
    category_path = DATASET_PATH / category

    # Paths
    normal_path = category_path / "NORMAL"
    pneumonia_path = category_path / "PNEUMONIA"

    # Check paths
    if not normal_path.exists() or not pneumonia_path.exists():
        raise FileNotFoundError("The specified category path does not exist.")

    # Find BACTERIA and VIRUS images inside PNEUMONIA
    bacteria_images = []
    virus_images = []
    normal_images = []

    for img_path in normal_path.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        normal_images.append(img_path)

    for img_path in pneumonia_path.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        if "bacteria" in img_path.name.lower():
            bacteria_images.append(img_path)
        elif "virus" in img_path.name.lower():
            virus_images.append(img_path)

    # Function to display images

    # Show images
    print("\nShowing NORMAL images...")
    display_images(normal_images, "NORMAL")

    print("\nShowing BACTERIA images...")
    display_images(bacteria_images, "BACTERIA")

    print("\nShowing VIRUS images...")
    display_images(virus_images, "VIRUS")


def display_plot(history:  Dict[Literal["accuracy", "val_accuracy", "loss", "val_loss"], List[float]], metric: Literal["accuracy", "loss"] = "accuracy"):
    """
    Display training and validation accuracy/loss plots.
    - `history`: Training history object
    - `metric`: Metric to plot (accuracy or loss)
    """
    if metric not in ["accuracy", "loss"]:
        raise ValueError("Metric must be either 'accuracy' or 'loss'.")

    plt.figure(figsize=(6, 4))
    plt.plot(history[metric], label=f"Train {metric.capitalize()}")
    plt.plot(history["val_" + metric],
             label=f"Validation {metric.capitalize()}")
    plt.title(f"{metric.capitalize()} over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.legend()

    plt.tight_layout()
    plt.show()


def show_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]):
    """
    Display confusion matrix.
    - `y_true`: True labels
    - `y_pred`: Predicted labels
    - `classes`: List of class names
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
