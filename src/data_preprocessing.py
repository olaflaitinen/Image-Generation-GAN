import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAIN_TEST_SPLIT_PATH
from utils import create_directory

def preprocess_data():
    create_directory(PROCESSED_DATA_PATH)
    create_directory(TRAIN_TEST_SPLIT_PATH)
    # Veri setini yükleme ve ön işleme kodu
    pass

if __name__ == "__main__":
    preprocess_data()
