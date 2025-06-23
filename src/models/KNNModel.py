from typing import Dict, List, Literal, Tuple, Union
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.common.SKModelInterface import SKModelInterface


class KNNModel(SKModelInterface):
    """ K-Nearest Neighbors model for chest X-ray classification.

    - Applies dimensionality reduction using PCA before classification
    - Uses standardization of features to improve KNN performance
    - Implements K-Nearest Neighbors classifier
    - Provides evaluation metrics compatible with CNN model

    The model is designed to classify chest X-rays into NORMAL, BACTERIA, and VIRUS categories.

    It is a sklearn KNeighborsClassifier so it needs to use the SKDataset (flattened images).
    """
    name = "KNN"

    def __init__(self, n_neighbors: int = 5, n_components: int = 100):
        """
        Initialize the KNN model.

        Args:
            n_neighbors (int): Number of neighbors to use for classification
            n_components (int): Number of PCA components for dimensionality reduction
        """
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.is_fitted = False
        self.num_classes = 3

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the KNN model.

        Args:
            x_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels in one-hot encoded format
        """
        # Convert one-hot encoded labels to class indices
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train_indices = np.argmax(y_train, axis=1)
        else:
            y_train_indices = y_train

        # Standardize features
        X_scaled = self.scaler.fit_transform(x_train)

        # Apply PCA for dimension reduction
        X_pca = self.pca.fit_transform(X_scaled)
        print(
            f"PCA explained variance ratio sum: {sum(self.pca.explained_variance_ratio_):.4f}")

        # Fit the KNN model
        self.model.fit(X_pca, y_train_indices)
        self.is_fitted = True

        self.model.predict(X_pca)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Predict classes for the given data.

        Args:
            x_test (np.ndarray): Test data

        Returns:
            np.ndarray: Predicted class indices for the test data
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        # Standardize features
        X_scaled = self.scaler.transform(x_test)

        # Apply PCA
        X_pca = self.pca.transform(X_scaled)

        # Return predictions
        return self.model.predict(X_pca)

    def grid_search(self, x_train: np.ndarray, y_train: np.ndarray,
                    x_val: np.ndarray, y_val: np.ndarray,
                    k_values: List[int] = [3, 5, 7, 9, 11]) -> Dict[str, int | float | np.floating]:
        """Perform grid search to find optimal hyperparameters.

        Args:
            x_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
            x_val (np.ndarray): Validation data
            y_val (np.ndarray): Validation labels
            k_values (List[int]): List of k values to try

        Returns:
            Dict[str, Union[int, float]]: Best parameters
        """
        # Convert labels to indices if needed
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train_indices = np.argmax(y_train, axis=1)
        else:
            y_train_indices = y_train

        if len(y_val.shape) > 1 and y_val.shape[1] > 1:
            y_val_indices = np.argmax(y_val, axis=1)
        else:
            y_val_indices = y_val

        # Standardize features
        X_train_scaled = self.scaler.fit_transform(x_train)
        X_val_scaled = self.scaler.transform(x_val)

        # Apply PCA
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)

        # Initialize variables to track best parameters
        best_accuracy = .0
        best_k = 0

        # Grid search over k values
        for k in k_values:
            # Create and fit KNN model
            temp_knn = KNeighborsClassifier(n_neighbors=k)
            temp_knn.fit(X_train_pca, y_train_indices)

            # Predict and calculate accuracy
            y_val_pred = temp_knn.predict(X_val_pca)
            accuracy = accuracy_score(y_val_indices, y_val_pred)

            print(f"K={k}, validation accuracy: {accuracy:.4f}")

            # Update best parameters if needed
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k

        # Update the model with best k
        self.n_neighbors = best_k
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        # Refit with best parameters
        self.model.fit(X_train_pca, y_train_indices)
        self.is_fitted = True

        print(f"Best K value: {best_k}, accuracy: {best_accuracy:.4f}")

        return {"n_neighbors": best_k, "accuracy": best_accuracy}

    def summary(self) -> None:
        """Print a summary of the KNN model configuration."""
        print(f"KNN Model Summary:")
        print(f"  - Number of neighbors (K): {self.n_neighbors}")
        print(f"  - PCA components: {self.n_components}")
        print(f"  - Number of classes: {self.num_classes}")
        if self.is_fitted:
            print(f"  - Model status: Fitted")
            print(
                f"  - PCA explained variance: {sum(self.pca.explained_variance_ratio_):.4f}")
        else:
            print(f"  - Model status: Not fitted")
