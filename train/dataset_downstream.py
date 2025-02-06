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
import torch.nn.functional as F

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
class Data_Loader(Dataset):
    def __init__(self, data_path, transform=None, create_overlap_flag=False):
        self.data_path = data_path
        with open(data_path, 'r') as json_file:
            self.data = json.load(json_file)
        self.root_path = os.path.dirname(self.data_path) + '/Dataset_cases_anonymization'
        self.transform = transform
        self.create_overlap_flag = create_overlap_flag

    def create_overlap(self, image, upper_mask, lower_mask, image_upper, image_lower, background):
        random_A = random.randint(2, 8)
        random_B = -random.randint(2, 8)

        width, height = image.shape
        translation_matrix_upper = np.float32([[1, 0, 0], [0, 1, random_A]])
        translation_matrix_lower = np.float32([[1, 0, 0], [0, 1, random_B]])

        upper_mask = cv2.warpAffine(upper_mask, translation_matrix_upper, (width, height))
        lower_mask = cv2.warpAffine(lower_mask, translation_matrix_lower, (width, height))
        image_upper = cv2.warpAffine(image_upper, translation_matrix_upper, (width, height))
        image_lower = cv2.warpAffine(image_lower, translation_matrix_lower, (width, height))

        mask_overlap = cv2.bitwise_and(upper_mask, lower_mask)

        overlap = background * mask_overlap
        non_zero_values = overlap[overlap != 0]
        if np.sum(mask_overlap) != 0:
            average_value = np.mean(non_zero_values)
            overlap = mask_overlap * average_value

        overlap_trans = 1 - overlap

        def edge_GaussianBlur(mask, image):
            image = (image * 255).astype(np.uint8)
            mask = (mask * 255).astype(np.uint8)

            edges = cv2.Canny(mask, 100, 200)

            kernel = np.ones((4, 4), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)

            blurred_edges = cv2.GaussianBlur(image, (3, 3), 0)

            blurred_mask = cv2.bitwise_and(blurred_edges, blurred_edges, mask=dilated_edges)

            inverse_mask = cv2.bitwise_not(dilated_edges)
            non_blurred_image = cv2.bitwise_and(image, image, mask=inverse_mask)

            final_image = cv2.add(blurred_mask, non_blurred_image)

            return final_image / 255

        image = 1 - ((1 - image_upper) * (1 - image_lower) / overlap_trans)

        final_image = edge_GaussianBlur(mask_overlap, image)
        # plt.subplot(1, 2, 1)
        # plt.imshow(image)
        # plt.subplot(1, 2, 2)
        # plt.imshow(final_image, 'gray')
        # plt.show()

        return final_image, upper_mask, lower_mask, image_upper, image_lower

    def input_image(self, target='', index=0):
        image_path = self.root_path + '/' + self.data[index][target]
        upper_mask_path = self.root_path + '/' + self.data[index]['{}_upper'.format(target)]
        lower_mask_path = self.root_path + '/' + self.data[index]['{}_lower'.format(target)]
        bcakground_root_path = self.root_path + '/background'
        background_path = bcakground_root_path + '/' + self.data[index][target].replace('/', '_')

        image = np.array(Image.open(image_path).convert("L"))
        background = np.array(Image.open(background_path).convert("L"))
        upper_mask = np.array(Image.open(upper_mask_path))
        lower_mask = np.array(Image.open(lower_mask_path))

        # Normalization to 0-1
        image = image / 255
        background = background / 255
        upper_mask[upper_mask != 0] = 1.0
        lower_mask[lower_mask != 0] = 1.0

        upper_mask = cv2.GaussianBlur(upper_mask * 255, (3, 3), 0)
        upper_mask[upper_mask != 0] = 1

        lower_mask = cv2.GaussianBlur(lower_mask * 255, (3, 3), 0)
        lower_mask[lower_mask != 0] = 1

        mask_all = cv2.bitwise_or(upper_mask, lower_mask)

        # plt.imshow(image * (1-mask_all))
        # plt.show()

        return image, upper_mask, lower_mask, background

    def create_overlap_standardisation(self, image, upper_mask, lower_mask, background):
        width, height = image.shape
        image_upper = image * upper_mask
        image_lower = image * lower_mask
        overlap_size = np.sum(cv2.bitwise_and(upper_mask, lower_mask))
        mask_all = cv2.bitwise_or(upper_mask, lower_mask)

        if self.create_overlap_flag:
            if overlap_size <= 20:
                range_size = 8
                image, upper_mask, lower_mask, image_upper, image_lower = self.create_overlap(image, upper_mask, lower_mask,
                                                                                              image_upper, image_lower,
                                                                                              background)
                fake_flag = 1
            else:
                range_size = 0
                image = image * mask_all
                fake_flag = 0
        else:
            range_size = 0
            image = image * mask_all
            fake_flag = 0

        upper_mask = upper_mask[range_size:width - range_size, range_size:height - range_size]
        lower_mask = lower_mask[range_size:width - range_size, range_size:height - range_size]

        upper_mask = cv2.resize(upper_mask, (width, height))
        lower_mask = cv2.resize(lower_mask, (width, height))

        image_upper = image_upper[range_size:width - range_size, range_size:height - range_size]
        image_lower = image_lower[range_size:width - range_size, range_size:height - range_size]
        image = image[range_size:width - range_size, range_size:height - range_size]

        return image, upper_mask, lower_mask, image_upper, image_lower

    def __getitem__(self, index):
        # Data path
        # moving
        moving_image, moving_upper_mask, moving_lower_mask, moving_background = self.input_image(target='moving',
                                                                                                 index=index)
        fixed_image, fixed_upper_mask, fixed_lower_mask, fixed_background = self.input_image(target='fixed',
                                                                                             index=index)

        # Create Overlap randomly
        moving_image, moving_upper_mask, moving_lower_mask, moving_image_upper, moving_image_lower = self.create_overlap_standardisation(
            moving_image,
            moving_upper_mask,
            moving_lower_mask,
            moving_background)
        fixed_image, fixed_upper_mask, fixed_lower_mask, fixed_image_upper, fixed_image_lower = self.create_overlap_standardisation(
            fixed_image,
            fixed_upper_mask,
            fixed_lower_mask,
            fixed_background)

        # Transform
        def to_tensor(image, image_upper, image_lower, lower_mask, upper_mask):
            org_image_upper = self.transform(Image.fromarray(image_upper))
            org_image_lower = self.transform(Image.fromarray(image_lower))
            image = self.transform(Image.fromarray(image))
            real_layer_images = torch.cat((org_image_lower, org_image_upper))

            upper_mask = torch.tensor(upper_mask).unsqueeze(0)
            lower_mask = torch.tensor(lower_mask).unsqueeze(0)

            layer_masks = torch.cat((lower_mask, upper_mask))

            image_cropping = torch.cat((image, image))
            image_cropping = image_cropping * layer_masks

            # mask cropping
            intersection_mask = torch.sum(layer_masks, dim=0).unsqueeze(0)
            intersection_mask[intersection_mask != 2] = 0
            intersection_mask = torch.cat((intersection_mask // 2, intersection_mask // 2))

            masks_cropping = layer_masks - intersection_mask
            # layer_masks_cropping[layer_masks_cropping != 1] = 0
            return image, layer_masks, image_cropping, masks_cropping, real_layer_images

        moving_image, moving_layer_masks, moving_image_cropping, moving_masks_cropping, moving_real_layer_images = to_tensor(
            moving_image, moving_image_upper, moving_image_lower, moving_lower_mask, moving_upper_mask)

        fixed_image, fixed_layer_masks, fixed_image_cropping, fixed_masks_cropping, fixed_real_layer_images = to_tensor(
            fixed_image, fixed_image_upper, fixed_image_lower, fixed_lower_mask, fixed_upper_mask)

        return moving_image_cropping, fixed_image_cropping, moving_layer_masks, fixed_layer_masks

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_path = ROOT_PATH + '/Data/DR_K1_train'
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    ])
    image_dataset = Data_Loader(data_path, transform, create_overlap_flag=False)
    data = torch.utils.data.DataLoader(dataset=image_dataset,
                                       batch_size=2,
                                       shuffle=True,
                                       drop_last=True
                                       )
    for moving_image_cropping, fixed_image_cropping, moving_layer_masks, fixed_layer_masks in data:
        plt.figure(figsize=(10, 2))
        plt.subplot(1, 5, 1)
        plt.imshow(moving_image_cropping.numpy()[0][1], 'gray')
        plt.subplot(1, 5, 2)
        plt.imshow(moving_layer_masks.numpy()[0][1])
        # plt.imshow(moving_image_cropping.numpy()[0][1], alpha=0.5)
        plt.subplot(1, 5, 3)
        plt.imshow(fixed_image_cropping.numpy()[0][1], 'gray')
        plt.subplot(1, 5, 4)
        plt.imshow(fixed_layer_masks.numpy()[0][1])
        plt.subplot(1, 5, 5)
        plt.tight_layout()
        plt.show()
    print(len(data))
