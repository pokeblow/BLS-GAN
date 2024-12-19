import torch.nn.functional as F
import torch
import torch.nn as nn
from backbones.resnet50 import ResNet
from backbones.denseunet import DenseUnet
from backbones.unet import Unet
from backbones.nestedunet import NestedUnet
from backbones.transunet import TransUnet
import timm
import torchvision.models as models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class Discriminator(nn.Module):
    def __init__(self, n_layers, backbone=None):
        super(Discriminator, self).__init__()
        self.backbone = backbone
        self.input_channel = n_layers
        self.output_channel = n_layers * 2

        if backbone == 'unet':
            self.backbone_network = Unet(in_channels=self.input_channel, out_channels=self.output_channel)
        if backbone == 'nestedunet':
            self.backbone_network = NestedUnet(num_channels=self.input_channel, num_class=self.output_channel)
        if backbone == 'deeplab':
            self.backbone_network = models.segmentation.deeplabv3_resnet50(pretrained=False)
            self.backbone_network.backbone.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone_network.classifier[4] = nn.Conv2d(256, self.output_channel, kernel_size=(1, 1), stride=(1, 1))
        if backbone == 'densenet':
            self.backbone_network = DenseUnet(img_ch=self.input_channel, output_ch=self.output_channel)
        if backbone == 'resnet':
            self.backbone_network = ResNet(in_channels=self.input_channel, out_channels=self.output_channel)
        if backbone == 'transunet':
            self.backbone_network = TransUnet(in_channels=self.input_channel, class_num=self.output_channel)



    def forward(self, x):
        x = self.backbone_network(x)
        if self.backbone == 'deeplab':
            x = x['out']
        output = F.sigmoid(x)

        return output
