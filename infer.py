# coding: utf-8
import os
import cv2

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import test_sots_root, test_hsts_root
from misc import check_mkdir, crf_refine
from model import ours
from datasets import ImageFolder2
from torch.utils.data import DataLoader
from torch import nn
from skimage.metrics import peak_signal_noise_ratio

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = '(ablation8 its) ours'
args = {
    'snapshot': 'iter_40000_loss_0.01221_lr_0.000000'
}

to_test = {'SOTS': test_sots_root}

to_pil = transforms.ToPILImage()


def main():
    net = ours().cuda()
    # net = nn.DataParallel(net)

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

    net.eval()
    psnrs = []

    # with torch.no_grad():
    #     for name, root in to_test.iteritems():
    #         dataset = ImageFolder2(root, 'test')
    #         dataloader = DataLoader(dataset, batch_size=8)
    #
    #         for idx, data in enumerate(dataloader):
    #             print 'predicting for %s: %d / %d' % (name, idx + 1, len(dataloader))
    #             check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
    #
    #             # haze_image, _, _, _, fs = data
    #             haze_image, fs = data
    #
    #             img_var = Variable(haze_image).cuda()
    #             res = net(img_var).data
    #             res[res > 1] = 1
    #             res[res < 0] = 0
    #
    #             for r, f in zip(res.cpu(), fs):
    #                 to_pil(r).save(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))
    for name, root in to_test.items():
        dataset = ImageFolder2(root, 'train')
        dataloader = DataLoader(dataset, batch_size=1)

        for idx, data in enumerate(dataloader):
            # haze_image, _, _, _, fs = data
            haze_image, gts, fs = data
            # print(haze_image.shape, gts.shape)

            print('predicting for %s [%s]: %d / %d' % (name, fs, idx + 1, len(dataloader)))
            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

            img_var = Variable(haze_image, volatile=True).cuda()
            res = net(img_var).data
            
            for i in range(len(fs)):
                r = res[i].cpu().numpy().transpose([1, 2, 0])
                gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                psnr = peak_signal_noise_ratio(gt, r)
                psnrs.append(psnr)

            for r, f in zip(res.cpu(), fs):
                to_pil(r).save(
                    os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))
    print(f"PSNR: {np.mean(psnrs):.6f}")


if __name__ == '__main__':
    main()
