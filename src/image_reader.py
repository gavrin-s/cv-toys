"""
An example of using multiprocessing for reading and processing images. I use it for caching datasets in RAM)
See timings in notebooks/image-reader.ipynb
"""
from typing import List
from multiprocessing.pool import ThreadPool

import numpy as np
import cv2


def read_images(image_names: List[str]) -> np.ndarray:
    """
    Read and resize multiply images in loop.
    """
    images = []
    for image_name in image_names:
        image = cv2.imread(image_name)[..., ::-1]
        image = cv2.resize(image, (224, 224))
        images.append(image)
    return np.array(images)


def read_images_multiprocessing(image_names: List[str], processes: int = 10) -> np.ndarray:
    """
    Read and resize multiply images in multiply processes.
    """
    count_images = len(image_names)
    group_size = count_images // processes + (count_images % processes > 0)
    image_names_by_groups = [image_names[group_size * i: group_size * (i + 1)] for i in range(processes)]
    results = ThreadPool(processes).map(read_images, image_names_by_groups)

    return np.concatenate([result for result in results if result.shape[0] > 0], axis=0)
