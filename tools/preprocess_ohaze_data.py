import os
from PIL import Image

from config import OHAZE_ROOT
from math import ceil
from tqdm import tqdm

if __name__ == '__main__':
    ohaze_root = OHAZE_ROOT
    crop_size = 512

    ori_root = os.path.join(ohaze_root, '# O-HAZY NTIRE 2018')
    ori_haze_root = os.path.join(ori_root, 'hazy')
    ori_gt_root = os.path.join(ori_root, 'GT')

    patch_root = os.path.join(ohaze_root, 'train_crop_{}'.format(crop_size))
    patch_haze_path = os.path.join(patch_root, 'hazy')
    patch_gt_path = os.path.join(patch_root, 'gt')

    os.makedirs(patch_root, exist_ok=True)
    os.makedirs(patch_haze_path, exist_ok=True)
    os.makedirs(patch_gt_path, exist_ok=True)

    # first 35 images for training
    train_list = [img_name for img_name in os.listdir(ori_haze_root)
                  if int(img_name.split('_')[0]) <= 35]

    for idx, img_name in enumerate(tqdm(train_list)):
        img_f_name, img_l_name = os.path.splitext(img_name)
        gt_f_name = '{}GT'.format(img_f_name[: -4])

        img = Image.open(os.path.join(ori_haze_root, img_name))
        gt = Image.open(os.path.join(ori_gt_root, gt_f_name + img_l_name))

        assert img.size == gt.size

        w, h = img.size
        stride = int(crop_size / 3.)
        h_steps = 1 + int(ceil(float(max(h - crop_size, 0)) / stride))
        w_steps = 1 + int(ceil(float(max(w - crop_size, 0)) / stride))

        for h_idx in range(h_steps):
            for w_idx in range(w_steps):
                ws0 = w_idx * stride
                ws1 = crop_size + ws0
                hs0 = h_idx * stride
                hs1 = crop_size + hs0
                if h_idx == h_steps - 1:
                    hs0, hs1 = max(h - crop_size, 0), h
                if w_idx == w_steps - 1:
                    ws0, ws1 = max(w - crop_size, 0), w
                img_crop = img.crop((ws0, hs0, ws1, hs1))
                gt_crop = gt.crop((ws0, hs0, ws1, hs1))

                img_crop.save(os.path.join(patch_haze_path, '{}_h_{}_w_{}.png'.format(img_f_name, h_idx, w_idx)))
                gt_crop.save(os.path.join(patch_gt_path, '{}_h_{}_w_{}.png'.format(gt_f_name, h_idx, w_idx)))
