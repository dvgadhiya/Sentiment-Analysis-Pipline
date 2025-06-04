import logging
from typing import Tuple

import numpy as np
import pandas as pd
from zenml import step
from typing_extensions import Annotated

from src.data_cleaning import DataCleaning, DataPreProcessingStrategyCNN, DataDivideStrategyCNN

@step
def clean_df_cnn(df: pd.DataFrame) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
    Annotated[dict, "word_index"]
]:
    """
    cleans the data and divides it into train and test sets

    Args:
        df: raw data

    Returns:
        X_train: Training data
        y_train: Testinggg data
        X_test: Training Labels
        y_test: Testing Labels
    """
    try:
        logging.info("Starting data Cleaning for CNN")
        process_strategy = DataPreProcessingStrategyCNN()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data, word_index = data_cleaning.handle_data()
        dive_strategy = DataDivideStrategyCNN()
        data_cleaning = DataCleaning(processed_data, dive_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data Cleaning Completed")
        return X_train, X_test, y_train, y_test, word_index
    except Exception as e:
        logging.error("Error in cleaning data".format(e))
        raise e
