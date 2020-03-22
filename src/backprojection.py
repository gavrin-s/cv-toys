import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.config import DATA_PATH


def get_mask(target_image: np.ndarray, object_image: np.ndarray) -> np.ndarray:
    """
    """
    # convert to hsv
    image_target_hsv = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV)
    image_object_hsv = cv2.cvtColor(object_image, cv2.COLOR_BGR2HSV)

    # get object's histogram
    histogram_object = cv2.calcHist([image_object_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(histogram_object, histogram_object, 0, 255, cv2.NORM_MINMAX)

    # get target mask
    image_mask = cv2.calcBackProject([image_target_hsv], [0, 1], histogram_object, [0, 180, 0, 256], 1)

    # postprocess
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_OPEN, kernel).astype(np.uint8)
    _, image_mask_threshold = cv2.threshold(image_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return image_mask_threshold


if __name__ == '__main__':
    image_target = cv2.imread(os.path.join(DATA_PATH, "palm.jpg"))
    image_object = cv2.imread(os.path.join(DATA_PATH, "palm_full.jpg"))

    mask = get_mask(image_target, image_object)

    mask3ch = cv2.merge([mask] * 3)
    masked_image = cv2.bitwise_and(image_target, mask3ch)

    result = np.hstack([image_target, mask3ch, masked_image])
    plt.imshow(result)
    plt.show()
