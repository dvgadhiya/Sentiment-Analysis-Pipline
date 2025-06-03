import logging
from abc import ABC, abstractmethod

import numpy as np
import tensorflow.keras.backend as K

class Evaluation(ABC):
    """
    abstract class defining the for evaluation our model
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the score between true labels and predicted labels
        Args:
            y_true (np.ndarray): true labels
            y_pred (np.ndarray): predicted labels
        Returns:
            None
        """
        pass

class MaskedAccuracy(Evaluation):
    """
    Evaluation class defining the for evaluation our model
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating masked accuracy")
            y_true = K.squeeze(y_true, axis=-1)  # remove last dim
            mask = K.cast(K.not_equal(y_true, 0), K.floatx())  # assuming 0 is the PAD token
            matches = K.cast(K.equal(K.cast(K.argmax(y_pred, axis=-1), K.floatx()), y_true), K.floatx())
            return K.sum(matches * mask) / K.sum(mask)
        except Exception as e:
            logging.error("Error while calculating masked accuracy: {}".format(e))
            raise e

class SequenceEvaluation(Evaluation):
    """
    Sequence-Level Accuracy
    How often is the entire sequence predicted correctly?
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating sequence accuracy")
            correct = 0
            total = len(y_true)
            for i in range(total):
                true_seq = y_true[i]
                pred_seq = y_pred[i].argmax(axis=-1)
                if np.array_equal(true_seq[true_seq != 0], pred_seq[true_seq != 0]):
                    correct += 1
            return correct / total
        except Exception as e:
            logging.error("Error while calculating sequence accuracy: {}".format(e))
            raise e

