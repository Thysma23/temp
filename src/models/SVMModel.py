from typing import Literal, Optional
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os
from sklearn.svm import SVC
from src.models.common.SKModelInterface import SKModelInterface


class SVMModel(SKModelInterface):
    """A class to represent a Support Vector Machine (SVM) model for classification tasks.

    This model uses the sklearn SVC (Support Vector Classification) implementation so it needs to use the SKDataset (flattened images)
    """
    model: SVC | LinearSVC
    name = "SVM"
    scaler: Optional[StandardScaler] = None
    pca: Optional[PCA] = None
    
    def __init__(self, C: float = 1.0, gamma: Literal['scale', 'auto'] | float = 'auto', 
                 kernel: Literal['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] = 'linear', 
                 use_pca: bool = True, pca_components: float = 0.95, 
                 cache_size: int = 2000, max_iter: int = 1000):
        # Use LinearSVC for linear kernel (much faster)
        if kernel == 'linear':
            self.model = LinearSVC(C=C, max_iter=max_iter, class_weight='balanced', dual=False)
        else:
            self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, 
                           class_weight='balanced', cache_size=cache_size, max_iter=max_iter)
        self.use_pca = use_pca
        self.pca_components = pca_components
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        # Ensure y_train is 1D array as expected by SVC
        if y_train.ndim > 1 and y_train.shape[1] == 1:
            y_train = y_train.ravel()
            
        # Feature scaling for better performance
        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x_train)
        
        # Dimensionality reduction with PCA
        if self.use_pca:
            self.pca = PCA(n_components=self.pca_components)
            x_train = self.pca.fit_transform(x_train)
            print(f"PCA reduced dimensions from {x_train.shape[1]} to {self.pca.n_components_}")
            
        # Train the model
        self.model.fit(x_train, y_train)
        
        # Save model to disk to avoid retraining
        os.makedirs('models_cache', exist_ok=True)
        joblib.dump(self, 'models_cache/svm_model.pkl')


    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if self.scaler:
            x_test = self.scaler.transform(x_test)
        
        if self.pca:
            x_test = self.pca.transform(x_test)
            
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict(x_test)
        else:
            # For LinearSVC which doesn't have predict_proba
            predictions = self.model.predict(x_test)
        return predictions
    
    def summary(self) -> None:
        print("SVM Model Summary:")
        params = self.model.get_params()
        for param, value in params.items():
            print(f"  {param}: {value}")
        if self.pca:
            print(f"PCA components: {self.pca.n_components_} ({self.pca_components*100}% variance)")