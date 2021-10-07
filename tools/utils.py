import os
import random


class AvgMeter(object):
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def random_crop(size, x):
    h, w = x.size()[2:]
    size_h = min(size, h)
    size_w = min(size, w)

    x1 = random.randint(0, w - size_w)
    y1 = random.randint(0, h - size_h)
    return x[:, :, y1: y1 + size_h, x1: x1 + size_w]
