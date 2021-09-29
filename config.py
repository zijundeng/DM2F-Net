# coding: utf-8
import os

root = os.path.dirname(os.path.abspath(__file__))

# # TestA
# TRAIN_A_ROOT = os.path.join(root, 'TrainA')
# TEST_A_ROOT = os.path.join(root, 'TestA')
# TEST_B_ROOT = os.path.join(root, 'nature')

# RESIDE
TRAIN_ITS_ROOT = os.path.join(root, 'data', 'RESIDE', 'ITS_v2')  # ITS
TEST_SOTS_ROOT = os.path.join(root, 'data', 'RESIDE', 'SOTS', 'nyuhaze500')  # SOTS indoor
# TEST_SOTS_ROOT = os.path.join(root, 'SOTS', 'outdoor')  # SOTS outdoor
# TEST_HSTS_ROOT = os.path.join(root, 'HSTS', 'synthetic')  # HSTS
