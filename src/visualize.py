import matplotlib.pyplot as plt
from model import build_generator
import numpy as np
import os
from config import RESULTS_PATH
from utils import create_directory

def visualize_results():
    create_directory(RESULTS_PATH)
    generator = build_generator()
    noise = np.random.normal(0, 1, (1, 100))
    generated_image = generator.predict(noise)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    plt.savefig(os.path.join(RESULTS_PATH, 'generated_image.png'))
    plt.show()

if __name__ == "__main__":
    visualize_results()
