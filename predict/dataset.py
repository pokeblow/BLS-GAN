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

ROOT_PATH = os.getcwd()

class Data_Loader(Dataset):
    def __init__(self, data_path, transform=None, model='nonoverlap'):
        '''
        :param model: dataset models for overlap, including 'nonoverlap', 'mix'
        '''
        self.data_path = data_path
        with open(data_path, 'r') as json_file:
            self.data = json.load(json_file)
        self.root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.data_path))))
        self.transform = transform
        self.model = model

    # def get_label(self):
    #     self.label = []
    #     upper_mask_path_list = x = self.df_list['upper']
    #     for i in upper_mask_path_list:
    #         upper_mask_path = self.root_path + '/' + i
    #         lower_mask_path = upper_mask_path.replace('upper', 'lower')
    #         upper_mask = np.array(Image.open(upper_mask_path)) / 255
    #         lower_mask = np.array(Image.open(lower_mask_path)) / 255
    #         overlap = upper_mask + lower_mask
    #         overlap[overlap != 2] = 0
    #         overlap_count = np.sum(overlap)
    #         if overlap_count == 0:
    #             self.label.append(0)
    #         else:
    #             self.label.append(1)
    #     return self.label

    def create_overlap(self, image, upper_mask, lower_mask, image_upper, image_lower):
        random_A = random.randint(2, 8)
        random_B = -random.randint(2, 8)

        width, height = image.shape
        translation_matrix_upper = np.float32([[1, 0, 0], [0, 1, random_A]])
        translation_matrix_lower = np.float32([[1, 0, 0], [0, 1, random_B]])

        upper_mask = cv2.warpAffine(upper_mask, translation_matrix_upper, (width, height))
        lower_mask = cv2.warpAffine(lower_mask, translation_matrix_lower, (width, height))
        image_upper = cv2.warpAffine(image_upper, translation_matrix_upper, (width, height))
        image_lower = cv2.warpAffine(image_lower, translation_matrix_lower, (width, height))

        image = 1 - ((1 - image_upper) * (1 - image_lower))

        return image, upper_mask, lower_mask, image_upper, image_lower

    def __getitem__(self, index):
        # Data path
        image_path = self.root_path + '/' + self.data[index]['image']
        background_path = image_path.replace('image', 'background')

        upper_mask_path = self.root_path + '/' + self.data[index]['upper']
        lower_mask_path = self.root_path + '/' + self.data[index]['lower']

        image = np.array(Image.open(image_path).convert("L"))
        background = np.array(Image.open(background_path).convert("L"))
        upper_mask = np.array(Image.open(upper_mask_path))
        lower_mask = np.array(Image.open(lower_mask_path))

        # Normalization to 0-1
        image = image / 255
        org_image = copy.deepcopy(image)
        background = background / 255


        upper_mask[upper_mask != 0] = 1.0
        lower_mask[lower_mask != 0] = 1.0

        width, height = image.shape

        mask_all = cv2.bitwise_or(upper_mask, lower_mask)

        # Create Overlap randomly
        image_upper = image * upper_mask
        image_lower = image * lower_mask
        overlap_size = np.sum(cv2.bitwise_and(upper_mask, lower_mask))

        if self.model == 'nonoverlap':
            range_size = 8
            image, upper_mask, lower_mask, image_upper, image_lower = self.create_overlap(image, upper_mask, lower_mask, image_upper, image_lower)
            fake_flag = 1
        if self.model == 'mix':
            if overlap_size <= 20:
                range_size = 8
                image, upper_mask, lower_mask, image_upper, image_lower = self.create_overlap(image, upper_mask, lower_mask, image_upper, image_lower)
                fake_flag = 1
            else:
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

        plt.subplot(1,4,1)
        plt.imshow(org_image, 'gray')
        plt.subplot(1,4,2)
        plt.imshow(image[0], 'gray')
        plt.subplot(1,4,3)
        plt.imshow(org_image * (1-mask_all), 'gray')
        plt.subplot(1,4,4)
        plt.imshow(background, 'gray')
        plt.show()

        return image, layer_masks, image_cropping, masks_cropping, real_layer_images, fake_flag


    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    data_path = ROOT_PATH + '/Data/Layer_seg_Dataset/partitions/mix/layer_seg_K1_train.json'
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    ])
    image_dataset = Data_Loader(data_path, transform, model='mix')
    data = torch.utils.data.DataLoader(dataset=image_dataset,
                                       batch_size=2,
                                       shuffle=True,
                                       drop_last=True
                                       )
    for image, layer_masks, image_cropping, masks_cropping, real_layer_images, fake_flag in data:
        print(fake_flag)
        plt.figure(figsize=(10,2))
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




