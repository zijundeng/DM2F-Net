# coding: utf-8
import os

import numpy as np
import torch
from torchvision import transforms

from config import TEST_SOTS_ROOT
from utils import check_mkdir
from model import DM2FNet
from datasets import SotsDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'RESIDE_ITS'
args = {
    'snapshot': 'iter_40000_loss_0.01256_lr_0.000000'
}

to_test = {'SOTS': TEST_SOTS_ROOT}

to_pil = transforms.ToPILImage()


def main():
    net = DM2FNet().cuda()
    # net = nn.DataParallel(net)

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

    net.eval()
    psnrs = []

    with torch.no_grad():
        for name, root in to_test.items():
            dataset = SotsDataset(root)
            dataloader = DataLoader(dataset, batch_size=1)

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze_image.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()
                res = net(haze).data

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    print('predicting for {} ({}/{}) [{}]: {:.4f}'.format(name, idx + 1, len(dataloader), fs[i], psnr))

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name,
                                     '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))
    print(f"PSNR for {name}: {np.mean(psnrs):.6f}")


if __name__ == '__main__':
    main()
