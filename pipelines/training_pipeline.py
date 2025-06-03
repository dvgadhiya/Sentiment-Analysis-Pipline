from zenml import pipeline
import logging
from src.evaluation import MaskedAccuracy
from steps.clean_data_cnn import clean_df_cnn
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df_lstm
from steps.evaluation import evaluate_model_cnn, evaluate_model_lstm
from steps.model_train import train_model_LSTM, train_model_CNN


@pipeline(enable_cache=True)
def training_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train,X_test,y_train,y_test,word_index = clean_df_lstm(df)
    model_1 = train_model_LSTM(X_train, y_train, word_index)
    masked_acc, sequence_acc = evaluate_model_lstm(model_1, X_test, y_test)
    X_train_CNN, X_test_CNN, y_train_CNN, y_test_CNN, word_index_CNN = clean_df_cnn(df)
    model_2 = train_model_CNN(X_train_CNN, y_train_CNN, word_index_CNN)
    model_2_accuracy = evaluate_model_cnn(model_2, X_test_CNN, y_test_CNN)


