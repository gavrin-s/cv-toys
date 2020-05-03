"""
See example in research/image-blending.ipynb
"""
import numpy as np


def blending_image(original, background, mask, gamma=0.):
    mask = np.atleast_3d(mask)
    if mask.max() > 1.:
        mask = mask / 255.

    result_image = (original * mask + background * (1.0 - mask) + gamma).astype(np.uint8)
    result_image = np.clip(result_image, 0, 255)
    return result_image
