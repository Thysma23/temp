from abc import ABC, abstractmethod
import numpy as np
from src.models.common.ModelInterface import ModelInterface


class SKModelInterface(ModelInterface, ABC):

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        pass
