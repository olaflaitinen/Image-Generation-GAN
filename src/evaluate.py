from model import build_generator, build_discriminator
import numpy as np
import os
from config import RESULTS_PATH
from utils import create_directory

def evaluate_gan():
    create_directory(RESULTS_PATH)
    generator = build_generator()
    discriminator = build_discriminator()
    # Modeli yükleme ve değerlendirme kodu
    pass

if __name__ == "__main__":
    evaluate_gan()
