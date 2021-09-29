# coding: utf-8
import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config import TRAIN_ITS_ROOT, TEST_SOTS_ROOT
from datasets import ITS, ImageFolder2
from misc import AvgMeter, check_mkdir
from model import ours
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cudnn.benchmark = True

torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = '(ablation8 its) ours'

args = {
    'iter_num': 40000,
    'train_batch_size': 16,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    'snapshot': '',
    'val_freq': 5000,
    'crop_size': 256
}

train_set = ITS(TRAIN_ITS_ROOT, True, args['crop_size'])
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=16, shuffle=True, drop_last=True)
val_set = ImageFolder2(TEST_SOTS_ROOT)
val_loader = DataLoader(val_set, batch_size=8)

criterion = nn.L1Loss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    net = ours().cuda().train()
    # net = nn.DataParallel(net)

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ])

    if len(args['snapshot']) > 0:
        print('training resumes from \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        train_loss_record = AvgMeter()
        loss_x_fusion_record, loss_x_phy_record, loss_x_p0_record, loss_x_p1_record, loss_x_p2_record, loss_x_p3_record, loss_t_record, loss_a_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for data in train_loader:
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            haze_image, gt_trans_map, gt_ato, gt, _ = data

            batch_size = haze_image.size(0)

            haze_image = Variable(haze_image).cuda()
            gt_trans_map = Variable(gt_trans_map).cuda()
            gt_ato = Variable(gt_ato).cuda()
            gt = Variable(gt).cuda()

            optimizer.zero_grad()

            x_fusion, x_phy, x_p0, x_p1, x_p2, x_p3, t, a = net(haze_image)

            loss_x_fusion = criterion(x_fusion, gt)
            loss_x_phy = criterion(x_phy, gt)
            loss_x_p0 = criterion(x_p0, gt)
            loss_x_p1 = criterion(x_p1, gt)
            loss_x_p2 = criterion(x_p2, gt)
            loss_x_p3 = criterion(x_p3, gt)

            loss_t = criterion(t, gt_trans_map)
            loss_a = criterion(a, gt_ato)

            loss = loss_x_fusion + loss_x_p0 + loss_x_p1 + loss_x_p2 + loss_x_p3 + loss_x_phy + 10 * loss_t + loss_a
            loss.backward()

            optimizer.step()

            train_loss_record.update(loss.data, batch_size)

            loss_x_fusion_record.update(loss_x_fusion.data, batch_size)
            loss_x_phy_record.update(loss_x_phy.data, batch_size)
            loss_x_p0_record.update(loss_x_p0.data, batch_size)
            loss_x_p1_record.update(loss_x_p1.data, batch_size)
            loss_x_p2_record.update(loss_x_p2.data, batch_size)
            loss_x_p3_record.update(loss_x_p3.data, batch_size)

            loss_t_record.update(loss_t.data, batch_size)
            loss_a_record.update(loss_a.data, batch_size)

            curr_iter += 1

            log = '[iter %d], [train loss %.5f], [loss_x_fusion %.5f], [loss_x_phy %.5f], [loss_x_p0 %.5f], ' \
                  '[loss_x_p1 %.5f], [loss_x_p2 %.5f], [loss_x_p3 %.5f], [loss_t %.5f], [loss_a %.5f], ' \
                  '[lr %.13f]' % \
                  (curr_iter, train_loss_record.avg, loss_x_fusion_record.avg, loss_x_phy_record.avg,
                   loss_x_p0_record.avg, loss_x_p1_record.avg, loss_x_p2_record.avg, loss_x_p3_record.avg,
                   loss_t_record.avg, loss_a_record.avg, optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % args['val_freq'] == 0:
                validate(net, curr_iter, optimizer)

            if curr_iter > args['iter_num']:
                return


def validate(net, curr_iter, optimizer):
    print('validating...')
    net.eval()

    loss_record = AvgMeter()

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            haze_image, gt, _ = data

            haze_image = Variable(haze_image).cuda()
            gt = Variable(gt).cuda()

            dehaze = net(haze_image)

            loss = criterion(dehaze, gt)
            loss_record.update(loss.data, haze_image.size(0))
    # for i, data in enumerate(val_loader):
    #     haze_image, gt, _ = data
    #
    #     haze_image = Variable(haze_image, volatile=True).cuda()
    #     gt = Variable(gt, volatile=True).cuda()
    #
    #     dehaze = net(haze_image)
    #
    #     loss = criterion(dehaze, gt)
    #     loss_record.update(loss.data, haze_image.size(0))

    snapshot_name = 'iter_%d_loss_%.5f_lr_%.6f' % (curr_iter + 1, loss_record.avg, optimizer.param_groups[1]['lr'])
    print('[validate]: [iter %d], [loss %.5f]' % (curr_iter + 1, loss_record.avg))
    torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    main()
