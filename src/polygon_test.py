"""
HW 1 https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html#exercises
"""
import os
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.config import DATA_PATH


def get_image_depth(size: Tuple[int, int], contour: np.ndarray) -> np.ndarray:
    """
    Get depth against contour
    """

    # get all distances
    matrix_distances = np.zeros(size, np.float32)  # for depth need only one chanel
    for i in range(size[0]):
        for j in range(size[1]):
            matrix_distances[i, j] = cv2.pointPolygonTest(contour, (j, i), True)

    # normalize distances
    mask_positive = matrix_distances > 0
    max_positive = matrix_distances[mask_positive].max()
    matrix_distances[mask_positive] = matrix_distances[mask_positive] / max_positive * 255

    mask_negative = matrix_distances < 0
    min_negative = matrix_distances[mask_negative].min()
    matrix_distances[mask_negative] = -1 * matrix_distances[mask_negative] / min_negative * 255

    close_to_zero = 5
    mask_contour = (matrix_distances >= -close_to_zero) & (matrix_distances <= close_to_zero)

    matrix_distances = matrix_distances.astype(np.int8)

    # create depth image
    image_depth = np.zeros(size + (3,), np.uint8)
    image_depth[mask_positive, 0] = 255 - matrix_distances[mask_positive]
    image_depth[mask_negative, 2] = 255 + matrix_distances[mask_negative]
    image_depth[mask_contour] = 255

    return image_depth


if __name__ == '__main__':
    image = cv2.imread(os.path.join(DATA_PATH, "apple.jpg"), 0)
    image_size = image.shape[:2]

    _, image_binary = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[2]

    image_depth = get_image_depth(image_size, contour)
    plt.imshow(image_depth)
    plt.show()
