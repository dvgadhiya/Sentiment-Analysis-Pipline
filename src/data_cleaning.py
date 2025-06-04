import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data.
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessingStrategy(DataStrategy):
    """
    Strategy for preprocessing data.
    """
    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Pre-process data.
        """
        try:
            maxlen = 100
            data = data.dropna()
            tokenized_texts, tokenized_labels = [], []
            for _, row in data.iterrows():
                tokens, labels = DataPreProcessingStrategy.get_token_labels(str(row["text"]), str(row["selected_text"]))
                tokenized_texts.append(tokens)
                tokenized_labels.append(labels)

            tokenizer = Tokenizer(oov_token="<OOV>")
            tokenizer.fit_on_texts([' '.join(t) for t in tokenized_texts])
            word_index = tokenizer.word_index
            label_encoder = LabelEncoder()
            label_encoder.fit(["O", "B", "I"])

            X1 = tokenizer.texts_to_sequences([' '.join(t) for t in tokenized_texts])
            y1 = [label_encoder.transform(lab) for lab in tokenized_labels]
            data.loc[:, "Text_sequences"] = list(pad_sequences(X1, maxlen=maxlen, padding='post'))
            data.loc[:, "Selected_text_sequences"] = list(pad_sequences(y1, maxlen=maxlen, padding='post'))
            return data,word_index
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e

    @staticmethod
    def get_token_labels(text: str, selected_text: str) -> Tuple[list,list]:
        """
        Generates Text and Selected text tokens
        Args:
            text:  str
            selected_text: str
        Returns:
            tokens : list
            labels : list
        """
        try:
            tokens = text.split()
            selected_tokens = selected_text.split()
            labels = ['O'] * len(tokens)
            for i in range(len(tokens)):
                if tokens[i:i + len(selected_tokens)] == selected_tokens:
                    labels[i] = 'B'
                    for j in range(1, len(selected_tokens)):
                        if i + j < len(tokens):
                            labels[i + j] = 'I'
                    break
            return tokens, labels
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e

class DataDivideStrategy(DataPreProcessingStrategy):
    """
    Strategy for divide data into train and test set.
    """
    def handle_data(self, data: pd.DataFrame):
        try:
            X = data["Text_sequences"]
            y = data["Selected_text_sequences"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e

class DataCleaning:
    """
    class for cleaning data which processes data and splits into train and test set.
    """

    def __init__(self,data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        """
        handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling cleaning: {}".format(e))
            raise e

class DataPreProcessingStrategyCNN(DataStrategy):
    """
    Data preprocessing strategy for CNN model.
    """
    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame,dict]:
        try:
            logging.info("Data preprocessing for CNN Started")
            tokenizer = Tokenizer(oov_token="<OOV>")
            tokenizer.fit_on_texts(data["selected_text"].astype(str))
            word_index = tokenizer.word_index
            tokenized_selected_text = tokenizer.texts_to_sequences(data["selected_text"].astype(str))
            tokenized_selected_text = pad_sequences(tokenized_selected_text, maxlen=30, padding="post")
            sent_encoder = LabelEncoder()
            encoded_sentiment = sent_encoder.fit_transform(data["sentiment"])
            encoded_sentiment = tf.keras.utils.to_categorical(encoded_sentiment)
            data.loc[:, "Tokenized_Selected_text"] = list(tokenized_selected_text)
            data.loc[:, "Encoded_Sentiment"] = list(encoded_sentiment)
            logging.info("Data preprocessing for CNN Completed")
            return data, word_index
        except Exception as e:
            logging.error("Error in preprocessing data for CNN: {}".format(e))
            raise e
class DataDivideStrategyCNN(DataPreProcessingStrategyCNN):
    """
    Strategy for divide data into train and test set.
    """
    def handle_data(self, data: pd.DataFrame) \
            -> Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
        try:
            logging.info("Data Dividing for CNN Started")
            X = data["Tokenized_Selected_text"]
            y = data["Encoded_Sentiment"]
            X_train: pd.DataFrame
            y_train: pd.Series
            X_test: pd.DataFrame
            y_test: pd.Series
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info("Data Dividing for CNN Completed")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e