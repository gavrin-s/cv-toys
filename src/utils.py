import time


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
