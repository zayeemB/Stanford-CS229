from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import numpy as np

import sys

MAX_ITER = 30

path_small = "./data/peppers-small.tiff"
path_large = "./data/peppers-large.tiff"

def main():

    small_image = imread(path_small)
    h, w, _ = small_image.shape

    rng = np.random.default_rng()
    means = rng.integers(1, 256, size=(16, 3))

    color = np.ones((h,w))

    iter = 0
    prev_means = np.zeros((16,3))

    while iter < MAX_ITER and np.sum(np.linalg.norm(prev_means - means, axis=1)) > 0:
        iter+=1
        diff = small_image[np.newaxis, :, :, :] - means[:, np.newaxis, np.newaxis, :]
        distances = np.linalg.norm(diff, axis=3)
        color = np.argmin(distances, axis=0)

        colors = np.arange(0,16)
        mask = (color[np.newaxis, :, :] == colors[:, np.newaxis, np.newaxis]).astype(int)
    
        prev_means = means.copy()
        means = np.einsum("jkl, ijk -> il", small_image, mask) / (np.sum(mask, axis=(1,2))[:, np.newaxis]+1)

    large_image = imread(path_large)
    diff = large_image[np.newaxis, :, :, :] - means[:, np.newaxis, np.newaxis, :]
    distances = np.linalg.norm(diff, axis=3)
    color = np.argmin(distances, axis=0)
    mean_replaced = means[color]

    final_image = mean_replaced.astype(np.uint8)

    plt.imshow(final_image)
    plt.imsave("peppers-large-compressed.png", final_image)
    # plt.show()
    
if __name__ == "__main__":
    main()
