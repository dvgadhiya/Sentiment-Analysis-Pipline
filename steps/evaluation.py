import logging
from typing import Tuple, Any

import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from zenml import step
from src.evaluation import MaskedAccuracy, SequenceEvaluation
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model

@step
def evaluate_model_lstm(
        model: Model,
        X_test: np.ndarray,
        y_test: np.ndarray,
) -> Tuple[float, float]:
    """
    Evaluates the model on ingested data.

    Args:
        model: Model,
        X_test: np.array,
        y_test: np.array,
    """
    try:
        X_test = np.array(X_test.tolist(), dtype=np.int32)
        y_test = np.array(y_test.tolist(), dtype=np.int32)

        logging.info("Getting model predictions...")
        y_pred = model.predict(X_test)

        # Convert predictions to class labels (from probabilities)
        y_pred_classes = np.argmax(y_pred, axis=-1)
        logging.info(f"y_pred_classes shape: {y_pred_classes.shape}")

        # Ensure y_test is in the right format
        if len(y_test.shape) == 3 and y_test.shape[-1] == 1:
            y_test = y_test.squeeze(-1)  # Remove last dimension if it's 1

        logging.info(f"Final y_test shape: {y_test.shape}")
        logging.info(f"Final y_pred_classes shape: {y_pred_classes.shape}")


        MaskedAccuracy_class = MaskedAccuracy()
        masked_acc = MaskedAccuracy_class.calculate_score(y_test,y_pred)
        logging.info(f"Masked accuracy: {masked_acc}")

        sequence_eval_class = SequenceEvaluation()
        sequence_eval_acc = sequence_eval_class.calculate_score(y_test,y_pred)
        logging.info(f"Sequence accuracy: {sequence_eval_acc}")

        return masked_acc, sequence_eval_acc
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e

@step
def evaluate_model_cnn(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
) -> float:
    """
    Evaluates the model on ingested data.

    Args:
        model: Model,
        X_test: np.array,
        y_test: np.array,
    """
    try:
        X_test = np.array(X_test.tolist(), dtype=np.int32)
        y_test = np.array(y_test.tolist(), dtype=np.int32)
        y_pred = model.predict(X_test)
        y_pred_onehot = y_pred.copy()  # Don't modify original

        for i in range(len(y_pred)):  # Fixed: range(len(y_pred)) instead of len(y_pred)
            k = np.argmax(y_pred[i])
            for j in range(len(y_pred[i])):
                y_pred_onehot[i][j] = 0
            y_pred_onehot[i][k] = 1

        accuracy = accuracy_score(y_test, y_pred_onehot)
        return accuracy
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e