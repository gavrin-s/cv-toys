import time
from typing import Tuple, Optional, Dict, Callable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Timer:
    """
    Example of using:
    with Timer() as t:
        sleep(2)

    >> Time: 2.00
    """
    def __init__(self, printing: bool = True):
        self.printing = printing
        self.start = 0
        self.end = 0
        self.interval = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.printing:
            print(f"Time: {self.interval:.2f} sec.")


def plot_images(images: Tuple[np.ndarray], names: Optional[Tuple[str]] = None, figsize: Optional[Tuple] = None,
                imshow_kwargs: Optional[Dict] = None):
    """
    Plot several images in row.
    """
    if names is None:
        names = [str(i) for i in range(len(images))]

    if imshow_kwargs is None:
        imshow_kwargs = dict()

    f, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    for i in range(len(images)):
        axes[i].imshow(images[i], **imshow_kwargs)
        axes[i].set_title(names[i])
        axes[i].axis("off")


def plot_statistics(values: Tuple[np.ndarray], function: Callable = sns.distplot, names: Optional[Tuple[str]] = None,
                    figsize: Optional[Tuple] = None):
    """
    Plot statistic function fow each value
    """
    if names is None:
        names = [str(i) for i in range(len(values))]

    f, axes = plt.subplots(1, len(values), figsize=figsize)
    # axes = np.atleast_2d(axes)
    for i in range(len(values)):
        function(values[i], ax=axes[i])
        axes[i].set_title(names[i])
