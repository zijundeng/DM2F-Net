import os
import math
import random
import torch


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


def sliding_forward(net, x: torch.Tensor, crop_size=1536):
    n, c, h, w = x.size()

    if h <= crop_size and w <= crop_size:
        return net(x)
    else:
        result = torch.zeros(n, c, h, w).cuda()
        count = torch.zeros(n, 1, h, w).cuda()
        stride = int(crop_size / 3.)

        h_steps = 1 + int(math.ceil(float(max(h - crop_size, 0)) / stride))
        w_steps = 1 + int(math.ceil(float(max(w - crop_size, 0)) / stride))

        for h_idx in range(h_steps):
            for w_idx in range(w_steps):
                ws0, ws1 = w_idx * stride, crop_size + w_idx * stride
                hs0, hs1 = h_idx * stride, crop_size + h_idx * stride
                if h_idx == h_steps - 1:
                    hs0, hs1 = max(h - crop_size, 0), h
                if w_idx == w_steps - 1:
                    ws0, ws1 = max(w - crop_size, 0), w
                result[:, :, hs0: hs1, ws0: ws1] += net(x[:, :, hs0: hs1, ws0: ws1]).data
                count[:, :, hs0: hs1, ws0: ws1] += 1
        assert torch.min(count) > 0
        result = result / count
        return result
