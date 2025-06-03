from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """Model Configs"""
    model_name_1: str = "BidirectionalLSTM"

class ModelNameConfigCNN(BaseModel):
    """Model Configs"""
    model_name_2: str = "TextCNN"

