import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.models.common.SKModelInterface import SKModelInterface


class ForestModel(SKModelInterface):
    """
    A class to represent a Random Forest model for classification tasks.

    Hyperparameters was chosen base on the grid search results

    It is a sklearn RandomForestClassifier so it needs to use the SKDataset (flattened images)
    """

    model: RandomForestClassifier
    name = "Random Forest"

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=150, random_state=42, min_samples_split=5, max_depth=30)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the Random Forest model on the training data.

        Args:
            x_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data labels.
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the test data.

        Args:
            x_test (np.ndarray): Test data features.

        Returns:
            np.ndarray: Predicted labels for the test data.
        """
        return self.model.predict(x_test)

    def summary(self) -> None:
        """
        Print the summary of the Random Forest model including its parameters.
        """
        print("Random Forest Model Summary:")
        params = self.model.get_params()
        for param, value in params.items():
            print(f"  {param}: {value}")

    def grid_search(self, x_train: np.ndarray, y_train: np.ndarray, param_grid: dict) -> None:
        """
        Perform grid search to find the best hyperparameters for the Random Forest model.

        The default parameter grid is defined within the method, but it can be overridden by passing a custom `param_grid`.
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        Args:
            x_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data labels.
            param_grid (dict): Dictionary with parameters names as keys and lists of parameter settings to try as values.
        """
        grid_search = GridSearchCV(
            estimator=self.model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(x_train, y_train)
        self.model = grid_search.best_estimator_
        print("Best parameters found: ", grid_search.best_params_)

        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        # Initialize the Random Forest Classifier
        rf_model = RandomForestClassifier(random_state=42)

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
        grid_search.fit(x_train, y_train)

        # Display the best parameters and best score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-Validation Score:", grid_search.best_score_)
