import logging
from typing import Any

import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from zenml import step
from src.model_dev import LSTM_Model, CNN_Model
from tensorflow.keras.models import Model

@step
def train_model_LSTM(
        X_train: np.ndarray,
        y_train: np.ndarray,
        word_index: dict,
) -> Model:
    """
    Trains models no ingested data.

    Args:
        X_train: pd.DataFrame,
        y_train: pd.Series,
        word_index: dict,
    """
    try:
        model = LSTM_Model()
        trained_model = model.train(X_train, y_train,word_index=word_index)
        return trained_model
    except Exception as e:
        logging.error("Error in training model: {} ".format(e))
        raise e

@step
def train_model_CNN(
        X_train: np.ndarray,
        y_train: np.ndarray,
        word_index: dict,
) -> Any:
    """
    Trains models no ingested data.

    Args:
        X_train: pd.DataFrame,
        y_train: pd.Series,
        word_index: dict,
    """
    try:
        model = CNN_Model()
        logging.info("Starting CNN model training...")
        trained_model = model.train(X_train, y_train, word_index=word_index)
        return trained_model
    except Exception as e:
        logging.error("Error in training model: {} ".format(e))
        raise e