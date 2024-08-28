from tensorflow.keras.optimizers import Adam
from model import build_generator, build_discriminator
import numpy as np
import os
from config import TRAIN_TEST_SPLIT_PATH, LOGS_PATH
from utils import create_directory

def train_gan():
    create_directory(LOGS_PATH)
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam())
    # Eğitim verilerini yükleme ve modeli eğitme kodu
    pass

if __name__ == "__main__":
    train_gan()
