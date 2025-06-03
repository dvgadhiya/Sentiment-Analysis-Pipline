import logging
from typing import Annotated
from typing import Tuple

import pandas as pd
from zenml import step
from src.evaluation import MaskedAccuracy, SequenceEvaluation
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model

@step
def evaluate_model_lstm(
        model: Model,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float,"masked_acc"],
    Annotated[float,"sequence_eval_acc"]
]:
    """
    Evaluates the model on ingested data.

    Args:
        df: the ingested data
    """
    try:
        y_pred = model.predict(X_test)
        MaskedAccuracy_class = MaskedAccuracy()
        masked_acc = MaskedAccuracy_class.calculate_score(y_test,y_pred)

        sequence_eval_class = SequenceEvaluation()
        sequence_eval_acc = sequence_eval_class.calculate_score(y_test,y_pred)

        return (masked_acc, sequence_eval_acc)
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e

@step
def evaluate_model_cnn(
        model: Model,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float,"accuracy"],
]:
    """
    Evaluates the model on ingested data.

    Args:
        df: the ingested data
    """
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e