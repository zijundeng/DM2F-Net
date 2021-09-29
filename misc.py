import numpy as np
import os
import torch
import random
from torch import nn
from torchvision import models
from config import vgg_path
import torch.nn.functional as F
from math import ceil
# import pydensecrf.densecrf as dcrf


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')


def sliding_forward(net, x, crop_size=2048):
    n, c, h, w = x.size()
    if h <= crop_size and w <= crop_size:
        return net(x)
    else:
        result = torch.zeros(n, 3, h, w).cuda()
        count = torch.zeros(n, 3, h, w).cuda()
        stride = int(crop_size / 3.)

        h_steps = 2 + max(h - crop_size, 0) / stride
        w_steps = 2 + max(w - crop_size, 0) / stride

        for h_idx in range(h_steps):
            for w_idx in range(w_steps):
                h_slice = slice(h_idx * stride, min(crop_size + h_idx * stride, h))
                w_slice = slice(w_idx * stride, min(crop_size + w_idx * stride, w))
                if h_idx == h_steps - 1:
                    h_slice = slice(max(h - crop_size, 0), h)
                if w_idx == w_steps - 1:
                    w_slice = slice(max(w - crop_size, 0), w)
                result[:, :, h_slice, w_slice] += net(x[:, :, h_slice, w_slice].contiguous())
                count[:, :, h_slice, w_slice] += 1
        assert torch.min(count) > 0
        result = result / count
        return result


def sliding_forward2(net, x, crop_size=(2048, 2048)):
    ch, cw = crop_size
    n, c, h, w = x.size()
    if h <= ch and w <= cw:
        return net(x)
    else:
        result = torch.zeros_like(x).cuda()
        count = torch.zeros_like(x).cuda()
        stride_h = int(ch / 3.)
        stride_w = int(cw / 3.)

        h_steps = 2 + max(h - ch, 0) / stride_h
        w_steps = 2 + max(w - cw, 0) / stride_w

        for h_idx in range(h_steps):
            for w_idx in range(w_steps):
                h_slice = slice(h_idx * stride_h, min(ch + h_idx * stride_h, h))
                w_slice = slice(w_idx * stride_w, min(cw + w_idx * stride_w, w))
                if h_idx == h_steps - 1:
                    h_slice = slice(max(h - ch, 0), h)
                if w_idx == w_steps - 1:
                    w_slice = slice(max(w - cw, 0), w)
                result[:, :, h_slice, w_slice] += net(x[:, :, h_slice, w_slice])
                count[:, :, h_slice, w_slice] += 1
        assert torch.min(count) > 0
        result = result / count
        return result


def sliding_forward3(net, x, shrink_factor, crop_size=1536):
    n, c, h, w = x.size()
    if h <= crop_size and w <= crop_size:
        x_sm = F.upsample(x, size=(int(h * shrink_factor), int(w * shrink_factor)), mode='bilinear')
        return net(x_sm, x)
    else:
        result = torch.zeros(n, c, h, w).cuda()
        count = torch.zeros(n, 1, h, w).cuda()
        stride = int(crop_size / 3.)

        h_steps = 1 + int(ceil(float(max(h - crop_size, 0)) / stride))
        w_steps = 1 + int(ceil(float(max(w - crop_size, 0)) / stride))

        for h_idx in range(h_steps):
            for w_idx in range(w_steps):
                ws0, ws1 = w_idx * stride, crop_size + w_idx * stride
                hs0, hs1 = h_idx * stride, crop_size + h_idx * stride
                if h_idx == h_steps - 1:
                    hs0, hs1 = max(h - crop_size, 0), h
                if w_idx == w_steps - 1:
                    ws0, ws1 = max(w - crop_size, 0), w
                x_patch = x[:, :, hs0: hs1, ws0: ws1]
                patch_h, patch_w = x_patch.size()[2:]
                x_patch_sm = F.upsample(x_patch, size=(int(patch_h * shrink_factor), int(patch_w * shrink_factor)), mode='bilinear')
                result[:, :, hs0: hs1, ws0: ws1] += net(x_patch_sm, x_patch)
                count[:, :, hs0: hs1, ws0: ws1] += 1
        assert torch.min(count) > 0
        result = result / count
        return result


def random_crop(size, x):
    h, w = x.size()[2:]
    size_h = min(size, h)
    size_w = min(size, w)

    x1 = random.randint(0, w - size_w)
    y1 = random.randint(0, h - size_h)
    return x[:, :, y1: y1 + size_h, x1: x1 + size_w]


class PerceptualLoss(nn.Module):
    def __init__(self, order):
        super(PerceptualLoss, self).__init__()
        assert order in [1, 2]
        vgg = models.vgg16()
        vgg.load_state_dict(torch.load(vgg_path))
        self.vgg = nn.Sequential(*(list(vgg.features.children())[: 9])).eval()

        self.criterion = nn.L1Loss() if order == 1 else nn.MSELoss()

        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)

        for m in self.vgg.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        return self.criterion(self.vgg(input), self.vgg(target).detach())
