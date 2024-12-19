import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os
import matplotlib.pyplot as plt

import cv2
 
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        bs = targets.size(0)
        smooth = 1
        
        probs = F.sigmoid(logits)
        m1 = probs.contiguous().view(bs, -1)
        m2 = targets.view(bs, -1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / bs
        return score


class LayerSegLoss_2(nn.Module):
    def __init__(self):
        super(LayerSegLoss_2, self).__init__()
        self.GS_criterion = nn.MSELoss()
        self.D_criterion = nn.BCELoss()
        self.D_criterion_dice = SoftDiceLoss()

    def forward(self, org_image, org_masks, pre_layer, recon_image, pre_masks):
        '''
        :param org_image: original image from dataset
        :param org_masks: original masks from dataset
        :param pre_layer: layer images from Generator
        :param recon_image: reconstructed images from Reconstructor
        :param pre_masks: discriminated masks from Discriminator with 4 channels
        :return: loss
        '''
        # Generation Similarity Loss
        org_mask_all = torch.sum(org_masks, 1).unsqueeze(1)
        org_mask_all[org_mask_all != 0] = 1

        org_image = org_image * org_mask_all
        recon_image = recon_image * org_mask_all

        GS_loss = torch.sqrt(self.GS_criterion(recon_image, org_image))

        # Discriminator Value Loss
        overlap_mask = torch.sum(org_masks, 1).unsqueeze(1) / 2
        overlap_mask[overlap_mask != 1] = 0

        overlap_label_layer = torch.cat((overlap_mask, overlap_mask, overlap_mask, overlap_mask), dim=1)

        label_layer = torch.cat((org_masks, org_masks), dim=1)

        D_loss_BCE = self.D_criterion(pre_masks, label_layer) + self.D_criterion(pre_masks * overlap_label_layer, overlap_label_layer)

        D_loss_Dice = self.D_criterion_dice(pre_masks, label_layer) + self.D_criterion_dice(pre_masks * overlap_label_layer, overlap_label_layer)

        D_loss = 0.5 * D_loss_BCE + 0.5 * D_loss_Dice

        print('---->', 'GS_Loss:', GS_loss.item(), "D_Loss:", D_loss.item())

        loss = 0.7 * D_loss + 0.3 * GS_loss

        return loss

class LayerSegLoss_1(nn.Module):
    def __init__(self):
        super(LayerSegLoss_1, self).__init__()
        self.GS_criterion = nn.MSELoss()
        self.D_criterion = nn.BCELoss()
        self.D_criterion_dice = SoftDiceLoss()

    def forward(self, org_image, org_masks, real_layer, pre_layer, recon_image, pre_masks):
        '''
        :param org_image: original image from dataset
        :param org_masks: original masks from dataset
        :param real_layer: original layer image from dataset (composite image)
        :param pre_layer: layer images from Generator
        :param recon_image: reconstructed images from Reconstructor
        :param pre_masks: discriminated masks from Discriminator with 4 channels
        :return: loss
        '''
        # Generation Similarity Loss
        org_mask_all = torch.sum(org_masks, 1).unsqueeze(1)
        org_mask_all[org_mask_all != 0] = 1

        org_image = org_image * org_mask_all
        recon_image = recon_image * org_mask_all

        GS_loss = torch.sqrt(self.GS_criterion(recon_image, org_image))

        # Layer image vs Ground truth (fake image input)
        LG_loss = torch.sqrt(self.GS_criterion(pre_layer, real_layer))

        # Discriminator Value Loss
        overlap_mask = torch.sum(org_masks, 1).unsqueeze(1) / 2
        overlap_mask[overlap_mask != 1] = 0

        overlap_label_layer = torch.cat((overlap_mask, overlap_mask, overlap_mask, overlap_mask), dim=1)

        label_layer = torch.cat((org_masks, org_masks), dim=1)

        D_loss_BCE = self.D_criterion(pre_masks, label_layer) + self.D_criterion(pre_masks * overlap_label_layer, overlap_label_layer)

        D_loss_Dice = self.D_criterion_dice(pre_masks, label_layer) + self.D_criterion_dice(pre_masks * overlap_label_layer, overlap_label_layer)

        D_loss = 0.5 * D_loss_BCE + 0.5 * D_loss_Dice
        
        print('---->', 'GS_Loss:', GS_loss.item(), "D_Loss:", D_loss.item(), "LG_Loss:", LG_loss.item())

        loss = 0.3 * D_loss + 0.3 * GS_loss + 0.4 * LG_loss

        return loss
