import logging
from abc import ABC, abstractmethod
import tensorflow as tf
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
        logging.info("Calculating masked accuracy")

        try:
            logging.info("Calculating masked accuracy")

            if isinstance(y_true, np.ndarray):
                y_true = tf.constant(y_true)
            if isinstance(y_pred, np.ndarray):
                y_pred = tf.constant(y_pred)

            if len(y_true.shape) == 3 and y_true.shape[-1] == 1:
                y_true = tf.squeeze(y_true, axis=-1)

            y_true = tf.cast(y_true, tf.int32)

            y_pred_classes = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

            matches = tf.cast(tf.equal(y_pred_classes, y_true), tf.float32)

            masked_matches = matches * mask
            total_valid_tokens = tf.reduce_sum(mask)

            accuracy = tf.cond(
                total_valid_tokens > 0,
                lambda: tf.reduce_sum(masked_matches) / total_valid_tokens,
                lambda: 0.0
            )

            return float(accuracy.numpy())

        except Exception as e:
            logging.error(f"Error while calculating masked accuracy: {e}")
            logging.error(f"y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
            logging.error(f"y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")
            raise e


class SequenceEvaluation(Evaluation):
    """
    Sequence-Level Accuracy
    How often is the entire sequence predicted correctly?
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating sequence accuracy")

            # Get predicted classes
            y_pred_classes = np.argmax(y_pred, axis=-1)

            # Remove last dimension from y_true if it exists
            if len(y_true.shape) > 2 and y_true.shape[-1] == 1:
                y_true = y_true.squeeze(-1)

            correct = 0
            total = len(y_true)

            for i in range(total):
                true_seq = y_true[i]
                pred_seq = y_pred_classes[i]

                # Only compare non-padding tokens (assuming 0 is padding)
                non_pad_mask = true_seq != 0
                true_non_pad = true_seq[non_pad_mask]
                pred_non_pad = pred_seq[non_pad_mask]

                if np.array_equal(true_non_pad, pred_non_pad):
                    correct += 1

            result = correct / total
            logging.info(f"Sequence accuracy result: {result}")
            return result

        except Exception as e:
            logging.error("Error while calculating sequence accuracy: {}".format(e))
            raise e
