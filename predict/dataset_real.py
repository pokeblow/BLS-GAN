import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
import cv2
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import pandas as pd
import copy
import random
import json
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import copy

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))


class Data_Loader(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        with open(data_path, 'r') as json_file:
            self.data = json.load(json_file)
        self.root_path = os.path.dirname(self.data_path)
        self.transform = transform

    def __getitem__(self, index):
        # Data path
        image_path = self.root_path + '/' + self.data[index]['image']
        upper_mask_path = self.root_path + '/' + self.data[index]['upper']
        lower_mask_path = self.root_path + '/' + self.data[index]['lower']
        background_path = self.root_path + '/' + self.data[index]['background']

        image = np.array(Image.open(image_path).convert("L"))
        background = np.array(Image.open(background_path).convert("L"))
        upper_mask = np.array(Image.open(upper_mask_path))
        lower_mask = np.array(Image.open(lower_mask_path))

        # Normalization to 0-1
        image = image / 255
        background = background / 255
        upper_mask[upper_mask != 0] = 1.0
        lower_mask[lower_mask != 0] = 1.0

        def gaussianBlur(mask):
            mask = cv2.GaussianBlur(mask * 255, (3, 3), 0)
            mask[mask != 0] = 1.0
            return mask

        upper_mask = gaussianBlur(upper_mask)
        lower_mask = gaussianBlur(lower_mask)

        width, height = image.shape

        mask_all = cv2.bitwise_or(upper_mask, lower_mask)

        # Create Overlap randomly
        image_upper = image * upper_mask
        image_lower = image * lower_mask

        range_size = 0
        image = image * mask_all
        fake_flag = 0

        # resize and cropping
        upper_mask = upper_mask[range_size:width - range_size, range_size:height - range_size]
        lower_mask = lower_mask[range_size:width - range_size, range_size:height - range_size]

        upper_mask = cv2.resize(upper_mask, (width, height))
        lower_mask = cv2.resize(lower_mask, (width, height))

        image_upper = image_upper[range_size:width - range_size, range_size:height - range_size]
        image_lower = image_lower[range_size:width - range_size, range_size:height - range_size]
        image = image[range_size:width - range_size, range_size:height - range_size]

        # Transform
        org_image_upper = self.transform(Image.fromarray(image_upper))
        org_image_lower = self.transform(Image.fromarray(image_lower))
        image = self.transform(Image.fromarray(image))

        upper_mask = torch.tensor(upper_mask).unsqueeze(0)
        lower_mask = torch.tensor(lower_mask).unsqueeze(0)

        layer_masks = torch.cat((lower_mask, upper_mask))
        real_layer_images = torch.cat((org_image_lower, org_image_upper))

        image_cropping = torch.cat((image, image))
        image_cropping = image_cropping * layer_masks

        # mask cropping
        intersection_mask = torch.sum(layer_masks, dim=0).unsqueeze(0)
        intersection_mask[intersection_mask != 2] = 0
        intersection_mask = torch.cat((intersection_mask // 2, intersection_mask // 2))

        masks_cropping = layer_masks - intersection_mask
        # layer_masks_cropping[layer_masks_cropping != 1] = 0

        name_list = self.data[index]['image'].split('/')
        name = name_list[1] + '_' + name_list[2]

        return image, layer_masks, image_cropping, masks_cropping, real_layer_images, name[:-4]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_path = ROOT_PATH + '/Data/LS_K1_overlap_train'
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    ])
    image_dataset = Data_Loader(data_path, transform)
    data = torch.utils.data.DataLoader(dataset=image_dataset,
                                       batch_size=2,
                                       shuffle=True,
                                       drop_last=True
                                       )
    for image, layer_masks, image_cropping, masks_cropping, real_layer_images, fake_flag in data:
        print(fake_flag)
        plt.figure(figsize=(10, 2))
        plt.subplot(1, 5, 1)
        plt.imshow(image.numpy()[0][0], 'gray')
        plt.subplot(1, 5, 2)
        plt.imshow(layer_masks.numpy()[0][0])
        plt.imshow(layer_masks.numpy()[0][1], alpha=0.5)
        plt.subplot(1, 5, 3)
        plt.imshow(real_layer_images.numpy()[0][0])
        plt.subplot(1, 5, 4)
        plt.imshow(image_cropping.numpy()[0][0])
        plt.subplot(1, 5, 5)
        plt.imshow(masks_cropping.numpy()[0][0])
        plt.tight_layout()
        plt.show()
    print(len(data))




