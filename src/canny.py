"""
Plays with Canny  Edge Detector
See tutorial https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
"""
from typing import Any, Optional
import os
import uuid
import cv2
import numpy as np

from src.config import DATA_PATH, TMP_PATH


def nothing(x: Any):
    pass


def main(image_source: np.ndarray, rescale: Optional[float] = None):
    cv2.namedWindow('Image')

    # create trackbars for color change
    cv2.createTrackbar('Threshold 1', 'Image', 0, 1000, nothing)
    cv2.createTrackbar('Threshold 2', 'Image', 0, 1000, nothing)
    cv2.createTrackbar('L2gradient', 'Image', 0, 1, nothing)

    image_egdes = cv2.Canny(image_source, 100, 200, L2gradient=True)

    while True:
        if rescale is not None:
            cv2.imshow('Image', cv2.resize(image_egdes, None, fx=rescale, fy=rescale))
        else:
            cv2.imshow('Image', image_egdes)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == 113:
            break
        elif k == 115:
            cv2.imwrite(os.path.join(TMP_PATH, f"{uuid.uuid4().hex}.jpg"), image_egdes)

        threshold1 = cv2.getTrackbarPos('Threshold 1', 'Image')
        threshold2 = cv2.getTrackbarPos('Threshold 2', 'Image')
        l2gradient = cv2.getTrackbarPos('L2gradient', 'Image')

        image_egdes = cv2.Canny(image_source, threshold1=threshold1, threshold2=threshold2,
                                L2gradient=l2gradient)

    cv2.destroyAllWindows()


# Use `ESC` or `q` for quit, `s` for save image in tmp folder
if __name__ == "__main__":
    image = cv2.imread(os.path.join(DATA_PATH, "likeaboss.jpg"))
    main(image, rescale=0.5)
