"""
Image histogram sample to show histogram
Refactored copy on https://github.com/opencv/opencv/blob/master/samples/python/color_histogram.py
"""

import os
from typing import Any
import numpy as np
import cv2

from src.config import DATA_PATH


def nothing(x: Any):
    pass


def main(image: np.ndarray):
    """
    """
    cv2.namedWindow("Image")
    cv2.namedWindow("Histogram")

    hsv_map = np.zeros((180, 256, 3), np.uint8)
    h, s = np.indices(hsv_map.shape[:2])
    hsv_map[:, :, 0] = h
    hsv_map[:, :, 1] = s
    hsv_map[:, :, 2] = 255
    hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)
    cv2.createTrackbar('Scale', 'Histogram', 10, 32, nothing)

    while True:
        cv2.imshow("Image", image)

        image_small = cv2.pyrDown(image)
        image_small_hsv = cv2.cvtColor(image_small, cv2.COLOR_BGR2HSV)
        mask_dark = image_small_hsv[..., 2] < 32
        image_small_hsv[mask_dark] = 0
        histogram = cv2.calcHist([image_small_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

        scale = cv2.getTrackbarPos('Scale', 'Histogram')

        histogram = np.clip(histogram * 0.005 * scale, 0, 1)
        histogram_bgr = hsv_map * histogram[:, :, np.newaxis] / 255.0
        cv2.imshow("Histogram", histogram_bgr)

        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == 113:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread(os.path.join(DATA_PATH, "palm.jpg"))
    main(image.copy())
