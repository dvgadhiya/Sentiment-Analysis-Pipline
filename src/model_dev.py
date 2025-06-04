import logging
from abc import ABC, abstractmethod
from typing import Union, Any

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense, GlobalAveragePooling1D
import numpy as np

class Model_abs(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, word_index: dict) -> Union[Model|Sequential| Any]:
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
            word_index: Tokenizer index
        Returns:
            None
        """
        pass

class LSTM_Model(Model_abs):
    """
    Time Distributed BiDirectional LSTM
    """
    def train(self, X_train: np.ndarray, y_train: np.ndarray, word_index: dict,**kwargs) -> Model:
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
            word_index: Tokenizer index
        Returns:
            Model
        """
        try:
            input1 = Input(shape=(100,))
            x = Embedding(len(word_index) + 1, 64)(input1)
            x = Bidirectional(LSTM(64, return_sequences=True))(x)
            output1 = TimeDistributed(Dense(3, activation='softmax'))(x)

            model1 = Model(input1, output1)
            model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            X_train = np.array(X_train.tolist(), dtype=np.int32)
            y_train = np.array(y_train.tolist(), dtype=np.int32)
            model1.fit(X_train, y_train[..., np.newaxis], epochs=5, batch_size=32, validation_split=0.1)
            logging.info("Model trained")
            return model1
        except Exception as e:
            logging.info("Error in model Training".format(e))
            raise e

class CNN_Model(Model_abs):
    """
    Textual CNN
    """
    def train(self, X_train: np.ndarray, y_train: np.ndarray,word_index: dict,**kwargs) -> Model:
        """
            Trains the model
            Args:
                X_train: Training data
                y_train: Training labels
                word_index: Tokenizer index
            Returns:
                Sequential
        """
        try:
            logging.info("Training CNN model")
            model2 = Sequential([
                Embedding(len(word_index) + 1, 32, input_length=30),
                GlobalAveragePooling1D(),
                Dense(32, activation='relu'),
                Dense(3, activation='softmax')
            ])
            model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            X_train = np.array(X_train.tolist(), dtype=np.int32)
            y_train = np.array(y_train.tolist(), dtype=np.int32)
            model2.fit(X_train, y_train, epochs=15, validation_split=0.1)
            logging.info("Training Completed")
            return model2

        except Exception as e:
            logging.info("Error in model Training".format(e))
            raise e