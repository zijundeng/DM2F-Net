import os
import os.path

import random
import numpy as np
from PIL import Image
import scipy.io as sio

import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import ToTensor

to_tensor = ToTensor()


def make_dataset(root):
    return [(os.path.join(root, 'hazy', img_name),
             os.path.join(root, 'trans', img_name),
             os.path.join(root, 'gt', img_name))
            for img_name in os.listdir(os.path.join(root, 'hazy'))]


def make_dataset_its(root):
    items = []
    for img_name in os.listdir(os.path.join(root, 'hazy')):
        idx0, idx1, ato = os.path.splitext(img_name)[0].split('_')
        gt = os.path.join(root, 'clear', idx0 + '.png')
        trans = os.path.join(root, 'trans', idx0 + '_' + idx1 + '.png')
        haze = os.path.join(root, 'hazy', img_name)
        items.append([haze, trans, float(ato), gt])

    return items


def make_dataset_ots(root):
    items = []
    for img_name in os.listdir(os.path.join(root, 'haze')):
        idx, _, _ = os.path.splitext(img_name)[0].split('_')
        gt = os.path.join(root, 'clear', idx + '.jpg')
        haze = os.path.join(root, 'haze', img_name)
        items.append([haze, gt])

    return items


def make_dataset_ohaze(root: str, mode: str):
    img_list = []
    for img_name in os.listdir(os.path.join(root, mode, 'hazy')):
        gt_name = img_name.replace('hazy', 'GT')
        assert os.path.exists(os.path.join(root, mode, 'gt', gt_name))
        img_list.append([os.path.join(root, mode, 'hazy', img_name),
                         os.path.join(root, mode, 'gt', gt_name)])
    return img_list


def make_dataset_oihaze_train(root, suffix):
    items = []
    for img_name in os.listdir(os.path.join(root, 'haze' + suffix)):
        gt = os.path.join(root, 'gt' + suffix, img_name)
        haze = os.path.join(root, 'haze' + suffix, img_name)
        items.append((haze, gt))

    return items


def make_dataset_oihaze_train_triple(root, suffix):
    items = []
    for img_name in os.listdir(os.path.join(root, 'haze' + suffix)):
        haze = os.path.join(root, 'haze' + suffix, img_name)
        gt = os.path.join(root, 'gt' + suffix, img_name)
        predict = os.path.join(root, 'predict' + suffix, img_name)
        items.append((haze, gt, predict))

    return items


def make_dataset_oihaze_test(root):
    items = []
    for img_name in os.listdir(os.path.join(root, 'haze')):
        img_f_name, img_l_name = os.path.splitext(img_name)
        gt_name = '%sGT%s' % (img_f_name[: -4], img_l_name)

        gt = os.path.join(root, 'gt', gt_name)
        haze = os.path.join(root, 'haze', img_name)

        items.append((haze, gt))

    return items


def random_crop(size, haze, gt, extra=None):
    w, h = haze.size
    assert haze.size == gt.size

    if w < size or h < size:
        haze = transforms.Resize(size)(haze)
        gt = transforms.Resize(size)(gt)
        w, h = haze.size

    x1 = random.randint(0, w - size)
    y1 = random.randint(0, h - size)

    _haze = haze.crop((x1, y1, x1 + size, y1 + size))
    _gt = gt.crop((x1, y1, x1 + size, y1 + size))

    if extra is None:
        return _haze, _gt
    else:
        # extra: trans or predict
        assert haze.size == extra.size
        _extra = extra.crop((x1, y1, x1 + size, y1 + size))
        return _haze, _gt, _extra


class ImageFolder(data.Dataset):
    def __init__(self, root, flip=False, crop=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.gt_ato_dict = sio.loadmat(os.path.join(root, 'ato.mat'))
        self.flip = flip
        self.crop = crop

    def __getitem__(self, index):
        haze_path, trans_path, gt_path = self.imgs[index]
        name = os.path.splitext(os.path.split(haze_path)[1])[0]

        haze = Image.open(haze_path).convert('RGB')
        trans = Image.open(trans_path).convert('L')
        gt = Image.open(gt_path).convert('RGB')

        assert haze.size == trans.size
        assert trans.size == gt.size

        if self.crop:
            haze, trans, gt = random_crop(self.crop, haze, trans, gt)

        if self.flip and random.random() < 0.5:
            haze = haze.transpose(Image.FLIP_LEFT_RIGHT)
            trans = trans.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        haze = to_tensor(haze)
        trans = to_tensor(trans)
        gt = to_tensor(gt)
        gt_ato = torch.Tensor([self.gt_ato_dict[name][0, 0]]).float()

        return haze, trans, gt_ato, gt, name

    def __len__(self):
        return len(self.imgs)


class ItsDataset(data.Dataset):
    """
    For RESIDE Indoor
    """

    def __init__(self, root, flip=False, crop=None):
        self.root = root
        self.imgs = make_dataset_its(root)
        self.flip = flip
        self.crop = crop

    def __getitem__(self, index):
        haze_path, trans_path, ato, gt_path = self.imgs[index]
        name = os.path.splitext(os.path.split(haze_path)[1])[0]

        haze = Image.open(haze_path).convert('RGB')
        trans = Image.open(trans_path).convert('L')
        gt = Image.open(gt_path).convert('RGB')

        assert haze.size == trans.size
        assert trans.size == gt.size

        if self.crop:
            haze, gt, trans = random_crop(self.crop, haze, gt, trans)

        if self.flip and random.random() < 0.5:
            haze = haze.transpose(Image.FLIP_LEFT_RIGHT)
            trans = trans.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        haze = to_tensor(haze)
        trans = to_tensor(trans)
        gt = to_tensor(gt)
        gt_ato = torch.Tensor([ato]).float()

        return haze, trans, gt_ato, gt, name

    def __len__(self):
        return len(self.imgs)


class OtsDataset(data.Dataset):
    """
    For RESIDE Outdoor
    """

    def __init__(self, root, flip=False, crop=None):
        self.root = root
        self.imgs = make_dataset_ots(root)
        self.flip = flip
        self.crop = crop

    def __getitem__(self, index):
        haze_path, gt_path = self.imgs[index]
        name = os.path.splitext(os.path.split(haze_path)[1])[0]

        haze = Image.open(haze_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        assert haze.size == gt.size

        if self.crop:
            haze, gt = random_crop(self.crop, haze, gt)

        if self.flip and random.random() < 0.5:
            haze = haze.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        haze = to_tensor(haze)
        gt = to_tensor(gt)

        return haze, gt, name

    def __len__(self):
        return len(self.imgs)


class SotsDataset(data.Dataset):
    def __init__(self, root, mode=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.mode = mode

    def __getitem__(self, index):
        haze_path, trans_path, gt_path = self.imgs[index]
        name = os.path.splitext(os.path.split(haze_path)[1])[0]

        haze = Image.open(haze_path).convert('RGB')
        haze = to_tensor(haze)

        idx0 = name.split('_')[0]
        gt = Image.open(os.path.join(self.root, 'gt', idx0 + '.png')).convert('RGB')
        gt = to_tensor(gt)
        if gt.shape != haze.shape:
            # crop the indoor images
            gt = gt[:, 10: 470, 10: 630]

        return haze, gt, name

    def __len__(self):
        return len(self.imgs)


class OHazeDataset(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.imgs = make_dataset_ohaze(root, mode)

    def __getitem__(self, index):
        haze_path, gt_path = self.imgs[index]
        name = os.path.splitext(os.path.split(haze_path)[1])[0]

        img = Image.open(haze_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        if 'train' in self.mode:
            # img, gt = random_crop(416, img, gt)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

            rotate_degree = np.random.choice([-90, 0, 90, 180])
            img, gt = img.rotate(rotate_degree, Image.BILINEAR), gt.rotate(rotate_degree, Image.BILINEAR)

        return to_tensor(img), to_tensor(gt), name

    def __len__(self):
        return len(self.imgs)


class OIHaze(data.Dataset):
    def __init__(self, root, mode, suffix=None, flip=False, crop=None):
        assert mode in ['train', 'test']
        self.root = root
        self.mode = mode
        if mode == 'train':
            self.img_name_list = make_dataset_oihaze_train(root, suffix)
        else:
            self.img_name_list = make_dataset_oihaze_test(root)

        self.flip = flip
        self.crop = crop

    def __getitem__(self, index):
        haze_path, gt_path = self.img_name_list[index]

        name = os.path.splitext(os.path.split(haze_path)[1])[0]
        haze = Image.open(haze_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        if self.crop:
            haze, gt = random_crop(self.crop, haze, gt)

        if self.flip and random.random() < 0.5:
            haze = haze.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        haze = to_tensor(haze)
        gt = to_tensor(gt)
        return haze, gt, name

    def __len__(self):
        return len(self.img_name_list)


class OIHaze5(data.Dataset):
    def __init__(self, root, flip=False, rotate=None, resize=1024):
        self.root = root
        self.img_name_list = make_dataset_oihaze_test(root)

        self.flip = flip
        self.rotate = rotate
        self.resize = transforms.Resize(resize)

    def __getitem__(self, index):
        haze_path, gt_path = self.img_name_list[index]

        name = os.path.splitext(os.path.split(haze_path)[1])[0]
        haze = Image.open(haze_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        if self.flip and random.random() < 0.5:
            haze = haze.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        if self.rotate:
            rotate_degree = random.random() * 2 * self.rotate - self.rotate
            haze = haze.rotate(rotate_degree, Image.BILINEAR)
            gt = gt.rotate(rotate_degree, Image.BILINEAR)

        haze_resize, gt_resize = self.resize(haze), self.resize(gt)

        haze, gt = to_tensor(haze), to_tensor(gt)
        haze_resize, gt_resize = to_tensor(haze_resize), to_tensor(gt_resize)
        return haze, gt, haze_resize, gt_resize, name

    def __len__(self):
        return len(self.img_name_list)


class OIHaze_T(data.Dataset):
    def __init__(self, root, mode, suffix=None, crop=None, flip=False, resize=1024):
        self.root = root
        assert mode in ['train', 'test']
        if mode == 'train':
            self.img_name_list = make_dataset_oihaze_train_triple(root, suffix)
        else:
            self.img_name_list = make_dataset_oihaze_test(root)
        self.mode = mode
        self.crop = crop
        self.flip = flip
        self.resize = transforms.Resize(resize)

    def __getitem__(self, index):
        if self.mode == 'train':
            haze_path, gt_path, predict_path = self.img_name_list[index]
            name = os.path.splitext(os.path.split(haze_path)[1])[0]
            haze = Image.open(haze_path).convert('RGB')
            gt = Image.open(gt_path).convert('RGB')
            predict = Image.open(predict_path).convert('RGB')

            if self.crop:
                haze, gt, predict = random_crop(self.crop, haze, gt, predict)

            if self.flip and random.random() < 0.5:
                haze = haze.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
                predict = predict.transpose(Image.FLIP_LEFT_RIGHT)

            haze, gt, predict = to_tensor(haze), to_tensor(gt), to_tensor(predict)
            return haze, gt, predict, name
        else:
            haze_path, gt_path = self.img_name_list[index]
            name = os.path.splitext(os.path.split(haze_path)[1])[0]
            haze = Image.open(haze_path).convert('RGB')
            gt = Image.open(gt_path).convert('RGB')

            haze_resize = self.resize(haze)
            haze, gt, haze_resize = to_tensor(haze), to_tensor(gt), to_tensor(haze_resize)
            return haze, gt, haze_resize, name

    def __len__(self):
        return len(self.img_name_list)


class OIHaze2(data.Dataset):
    def __init__(self, root, mode, suffix=None, flip=False, crop=None, scale=None, rotate=None):
        assert mode in ['train', 'test']
        self.root = root
        self.mode = mode
        if mode == 'train':
            self.img_name_list = make_dataset_oihaze_train(root, suffix)
        else:
            self.img_name_list = make_dataset_oihaze_test(root)
            self.scale = transforms.Resize(scale)

        self.flip = flip
        self.crop = crop
        self.rotate = rotate

    def __getitem__(self, index):
        haze_path, gt_path = self.img_name_list[index]

        name = os.path.splitext(os.path.split(haze_path)[1])[0]
        haze = Image.open(haze_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        if self.mode == 'test':
            haze_lr = self.scale(haze)
            haze_lr = to_tensor(haze_lr)
        else:
            if self.rotate:
                rotate_degree = random.random() * 2 * self.rotate - self.rotate
                haze = haze.rotate(rotate_degree, Image.BILINEAR)
                gt = gt.rotate(rotate_degree, Image.BILINEAR)

            if self.crop:
                haze, gt = random_crop(self.crop, haze, gt)

            if self.flip and random.random() < 0.5:
                haze = haze.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        haze = to_tensor(haze)
        gt = to_tensor(gt)

        if self.mode == 'test':
            return haze, gt, haze_lr, name
        else:
            return haze, gt, name

    def __len__(self):
        return len(self.img_name_list)


class OIHaze2_2(data.Dataset):
    def __init__(self, root, mode, flip=False, crop=None):
        assert mode in ['train', 'test']
        self.root = root
        self.mode = mode
        if mode == 'train':
            self.img_name_list = make_dataset_oihaze_train(root)
        else:
            self.img_name_list = make_dataset_oihaze_test(root)

        self.flip = flip
        self.crop = crop

    def __getitem__(self, index):
        haze_path, gt_path = self.img_name_list[index]

        name = os.path.splitext(os.path.split(haze_path)[1])[0]
        haze = Image.open(haze_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        if self.mode == 'test':
            haze_lr = haze.resize((1024, 1024), resample=Image.BILINEAR)
            haze_lr = to_tensor(haze_lr)

        if self.crop:
            haze, gt = random_crop(self.crop, haze, gt)

        haze = to_tensor(haze)
        gt = to_tensor(gt)

        if self.mode == 'test':
            return haze, gt, haze_lr, name
        else:
            return haze, gt, name

    def __len__(self):
        return len(self.img_name_list)


class OIHaze4(data.Dataset):
    def __init__(self, root, mode, crop=None):
        assert mode in ['train', 'test']
        self.root = root
        self.mode = mode
        if mode == 'train':
            self.img_name_list = make_dataset_oihaze_train(root)
        else:
            self.img_name_list = make_dataset_oihaze_test(root)

        self.crop = crop

    def __getitem__(self, index):
        haze_path, gt_path = self.img_name_list[index]

        name = os.path.splitext(os.path.split(haze_path)[1])[0]
        haze = Image.open(haze_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        if self.mode == 'train':
            if self.crop:
                haze, gt = random_crop(self.crop, haze, gt)
        else:
            haze_512 = to_tensor(transforms.Resize(512)(haze))
            haze_1024 = to_tensor(transforms.Resize(1024)(haze))
            haze_2048 = to_tensor(transforms.Resize(2048)(haze))

        haze = to_tensor(haze)
        gt = to_tensor(gt)

        if self.mode == 'train':
            return haze, gt, name
        else:
            return haze, gt, haze_512, haze_1024, haze_2048, name

    def __len__(self):
        return len(self.img_name_list)


class OIHaze3(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.img_name_list = make_dataset_oihaze_test(root)

    def __getitem__(self, index):
        haze_path, gt_path = self.img_name_list[index]

        name = os.path.splitext(os.path.split(haze_path)[1])[0]
        haze = Image.open(haze_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        resize = transforms.Resize(512)
        haze_lr = resize(haze)
        haze_lr = to_tensor(haze_lr)

        haze = to_tensor(haze)
        gt = to_tensor(gt)
        return haze, gt, haze_lr, name

    def __len__(self):
        return len(self.img_name_list)


class ImageFolder3(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = [os.path.join(root, img_name) for img_name in os.listdir(root)]

    def __getitem__(self, index):
        haze_path = self.imgs[index]
        name = os.path.splitext(os.path.split(haze_path)[1])[0]

        haze = Image.open(haze_path).convert('RGB')
        haze = to_tensor(haze)
        return haze, name

    def __len__(self):
        return len(self.imgs)
