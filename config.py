# coding: utf-8
import os

root = '/home/jqxu/data/RESIDE'
train_a_root = os.path.join(root, 'TrainA')
test_a_root = os.path.join(root, 'TestA')
test_b_root = os.path.join(root, 'nature')

train_its_root = os.path.join(root, 'ITS_v2')
test_hsts_root = os.path.join(root, 'HSTS', 'synthetic', '')
# test_sots_root = os.path.join(root, 'SOTS', 'nyuhaze500')
test_sots_root = os.path.join(root, 'SOTS', 'outdoor')

ohaze_ori_root = os.path.join(root, 'O-HAZE')
train_ohaze_ori_root = os.path.join(ohaze_ori_root, 'train')
test_ohaze_ori_root = os.path.join(ohaze_ori_root, 'test')

ohaze_resize_root = os.path.join('/media/b3-542/4edbfae9-f11c-447b-b07c-585bc4017092/DataSets/dehaze', 'O-HAZE')
train_ohaze_resize_root = os.path.join(ohaze_resize_root, 'train')
test_ohaze_resize_root = os.path.join(ohaze_resize_root, 'test')

ihaze_resize_root = os.path.join('/media/b3-542/4edbfae9-f11c-447b-b07c-585bc4017092/DataSets/dehaze', 'I-HAZE')
train_ihaze_resize_root = os.path.join(ihaze_resize_root, 'train')
test_ihaze_resize_root = os.path.join(ihaze_resize_root, 'test')

ohaze_hd_root = os.path.join('/media/b3-542/4edbfae9-f11c-447b-b07c-585bc4017092/DataSets/dehaze', 'O-HAZE-HD')
train_ohaze_hd_root = os.path.join(ohaze_hd_root, 'train')
test_ohaze_hd_root = os.path.join(ohaze_hd_root, 'test')

ihaze_hd_root = os.path.join('/media/b3-542/4edbfae9-f11c-447b-b07c-585bc4017092/DataSets/dehaze', 'I-HAZE-HD')
train_ihaze_hd_root = os.path.join(ihaze_hd_root, 'train')
test_ihaze_hd_root = os.path.join(ihaze_hd_root, 'test')

ihaze_root = os.path.join('/media/b3-542/4edbfae9-f11c-447b-b07c-585bc4017092/DataSets/dehaze', 'I-HAZE')
train_ihaze_root = os.path.join(ihaze_root, 'train')
test_ihaze_root = os.path.join(ihaze_root, 'test')

train_ots_root = '/media/b3-542/4edbfae9-f11c-447b-b07c-585bc4017092/DataSets/dehaze/OTS'
test_natural2_root = '/home/b3-542/文档/DataSets/dehaze/natural2'

vgg_path = '/media/b3-542/454BAA0333169FE1/Packages/PyTorch Pretrained/VggNet/vgg16-397923af.pth'
