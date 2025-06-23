from typing import Dict, List, Literal
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.api.optimizers import Adam
import numpy as np
from sklearn.utils import compute_class_weight
import tensorflow as tf
from config import RESIZE_DIM, IS_DATA_AUGMENTED
from src.models.common.ModelInterface import ModelInterface


class CNNModel(ModelInterface):
    """ Defines the neural network architecture and prepares it for training.

    - Sets up data augmentation to artificially expand the training dataset.
    - Creates a multi-layer CNN with 4 convolutional blocks (32→64→128→256 filters).
    - Uses BatchNormalization and Dropout layers to prevent overfitting.
    - Compiles the model with Adam optimizer and categorical cross-entropy loss.
    - Calculates class weights to handle class imbalance in the dataset.

    The model is designed to classify chest X-rays into NORMAL, BACTERIA, and VIRUS categories using the grayscales images.

    It is a Keras Sequential model so it needs to use the TFDataset (3 dimensions).
    """

    model: Sequential
    name = "CNNModel"

    def __init__(self):
        """
        Initializes the CNN model.

        """
        self.input_shape = RESIZE_DIM + \
            (1,)  # Adding channel dimension for grayscale images
        self.num_classes = 3
        self.model = self.__create_cnn_model()

    def __create_cnn_model(self) -> Sequential:
        """ Define the CNN architecture.

        Returns:
            Sequential: The compiled CNN model.
        """
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu',
                   padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Fourth convolutional block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Flatten and fully connected layers
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            # Output layer with softmax for classification
            Dense(self.num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),  # type: ignore
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def __get_callbacks(self) -> List:
        """ Define callbacks for better training performance.

        Returns:
            list: A list of callbacks including EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau.
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        )

        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001,
            verbose=1
        )

        return [early_stopping, checkpoint, reduce_lr]

    def __get_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """ Calculate class weights to handle imbalance.

        Args:
            y_train (np.ndarray): Training labels in one-hot encoded format.

        Returns:
            dict: A dictionary mapping class indices to their respective weights.
        """
        y_integers = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_integers),
            y=y_integers
        )
        return dict(enumerate(class_weights))

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 30) -> Dict[Literal["accuracy", "val_accuracy", "loss", "val_loss"], List[float]]:
        """ Train the model with data augmentation and class weights.

        Args:
            x_train (np.ndarray): Training data.
            y_train (np.ndarray): Training labels in one-hot encoded format.
            x_val (np.ndarray): Validation data.
            y_val (np.ndarray): Validation labels in one-hot encoded format.
            epochs (int): Number of training epochs (default: 30).

        Returns:
            dict: Training history containing metrics like accuracy and loss.
        """

        print(x_train.shape)
        # Get callbacks
        callbacks = self.__get_callbacks()

        # Calculate class weights
        class_weights = self.__get_class_weights(y_train)

        # Train the model
        train = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        return train.history

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """ Predict the model.

        Args:
            x_test (np.ndarray): Test data.

        Returns:
            np.ndarray: Predicted class indices for the test data.
        """
        predictions = self.model.predict(x_test)
        return np.argmax(predictions, axis=1)

    def summary(self) -> None:
        """ Prints the summary of the model. """
        self.model.summary()
