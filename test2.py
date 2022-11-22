import os
import sys
sys.path.append('/workspace/gan/3D-FM-GAN/')
from PIL import Image
import torch
import torch.nn as nn
from model.dataset import ReconDataset, DisentangleDataset
from torch.utils.data import DataLoader


root = '/workspace/gan/3D-FM-GAN/data'
rds = ReconDataset(root=root, sub_dir='ffhq', random_flip=True, normalize=True)
rdl = DataLoader(rds, batch_size=32, shuffle=True)

test = torch.randn((10, 512, 512))
for i, data in enumerate(rdl):
    p_img = data['p1'][0]
    r_img = data['r1'][0]
    test2 = test[:3]
    print('breakpoint')


# import numpy as np
# color = np.full((500,500,3), (255, 0, 0), np.uint8)

# print('breakpoint')