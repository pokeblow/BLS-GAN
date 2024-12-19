'''
version: 1.0
data: 2023.3.8
model: FCN
'''

import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from backbones.vgg11 import VGG11


class Reconstructor(nn.Module):
    def __init__(self):
        super(Reconstructor, self).__init__()
        self.if_CP = True

        self.model = VGG11(in_channels=1)

        self.liner = nn.Sequential(
            nn.Linear(32768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def reconstruct(self, org, k, masks):
        # image
        upper_image = org[:, 1, :, :].unsqueeze(1)
        lower_image = org[:, 0, :, :].unsqueeze(1)

        upper_trans = 1 - upper_image
        lower_trans = 1 - lower_image

        org_mask_and = torch.sum(masks, 1).unsqueeze(1)
        org_mask_and[org_mask_and != 2] = 0
        and_trans = (org_mask_and / 2) * k

        and_trans = 1 - and_trans

        if self.if_CP:
            output = (1 - (upper_trans * lower_trans / and_trans))
        else:
            output = (1 - (upper_trans * lower_trans))

        return output

    def set_correct_parameter(self, if_CP=True):
        self.if_CP = if_CP

    def forward(self, image, org, masks):
        # 编码器
        x = torch.clone(image)

        x = self.model(x)

        x = x.view(x.size(0), -1)
        x = self.liner(x)

        _, _, h, w = image.shape
        batch_size, _ = x.shape

        k = x.view(batch_size, 1, 1, 1)
        k = k.expand(batch_size, 1, h, w)

        recon_image = self.reconstruct(org, k, masks)

        return recon_image, x
