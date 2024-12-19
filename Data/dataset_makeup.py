import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import json
import copy
from pycocotools.coco import COCO

import pandas as pd
import argparse

ROOT_PATH = os.getcwd()


def makeup_data(cross_list, file_path, cross, trainortest=''):
    root_path = 'Dataset_cases'

    file_list_nonoverlap = []
    file_list_overlap = []
    file_list = []

    for case in cross_list:
        overlap_count = 0
        phase_list_org = os.listdir(root_path + '/' + case)
        phase_list_check = [file for file in phase_list_org if '.bmp' not in file]
        for image in phase_list_check:
            # all images
            row_dic = {'image': '', 'upper': '', 'lower': '', 'background': ''}
            row_dic['image'] = root_path + '/' + case + '/' + image
            row_dic['upper'] = root_path + '/' + case + '/' + image[:-4] + '_mask_upper.bmp'
            row_dic['lower'] = root_path + '/' + case + '/' + image[:-4] + '_mask_lower.bmp'
            row_dic['background'] = root_path + '/background/' + '{}_{}'.format(case, image)
            background = np.array(Image.open(root_path + '/background/' + '{}_{}'.format(case, image)))
            file_list.append(row_dic)

            upper_mask = np.array(
                Image.open(ROOT_PATH + '/' + root_path + '/' + case + '/' + image[:-4] + '_mask_upper.bmp'))
            lower_mask = np.array(
                Image.open(ROOT_PATH + '/' + root_path + '/' + case + '/' + image[:-4] + '_mask_lower.bmp'))
            upper_mask[upper_mask != 0] = 1.0
            lower_mask[lower_mask != 0] = 1.0
            overlap_size = np.sum(cv2.bitwise_and(upper_mask, lower_mask))

            if overlap_size >= 20:
                # overlap images
                row_dic = {'image': '', 'upper': '', 'lower': '', 'background': ''}
                row_dic['image'] = root_path + '/' + case + '/' + image
                row_dic['upper'] = root_path + '/' + case + '/' + image[:-4] + '_mask_upper.bmp'
                row_dic['lower'] = root_path + '/' + case + '/' + image[:-4] + '_mask_lower.bmp'
                row_dic['background'] = root_path + '/background/' + '{}_{}'.format(case, image)
                file_list_overlap.append(row_dic)
            else:
                # non-overlap images
                row_dic = {'image': '', 'upper': '', 'lower': '', 'background': ''}
                row_dic['image'] = root_path + '/' + case + '/' + image
                row_dic['upper'] = root_path + '/' + case + '/' + image[:-4] + '_mask_upper.bmp'
                row_dic['lower'] = root_path + '/' + case + '/' + image[:-4] + '_mask_lower.bmp'
                row_dic['background'] = root_path + '/background/' + '{}_{}'.format(case, image)
                file_list_nonoverlap.append(row_dic)

    # 将数据写入.json文件
    save_path = '{}/LS_{}_all_{}'.format(file_path, cross, trainortest)
    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(file_list, json_file, ensure_ascii=False, indent=4)

    save_path = '{}/LS_{}_overlap_{}'.format(file_path, cross, trainortest)
    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(file_list_overlap, json_file, ensure_ascii=False, indent=4)

    save_path = '{}/LS_{}_nonoverlap_{}'.format(file_path, cross, trainortest)
    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(file_list_nonoverlap, json_file, ensure_ascii=False, indent=4)

    json_list = []
    for joint in cross_list:
        phase_list_org = os.listdir(root_path + '/' + joint)
        phase_list_check = [file for file in phase_list_org if '.bmp' not in file]
        phase_list = []
        for phase in phase_list_check:
            if phase[:-4] + '_mask_upper.bmp' in phase_list_org:
                if phase[:-4] + '_mask_lower.bmp' in phase_list_org:
                    phase_list.append(phase)
                else:
                    print(phase)
            else:
                print(phase)
        row_list = {'joint_name': '', 'moving': '', 'moving_upper': '', 'moving_lower': '',
                    'fixed': '', 'fixed_upper': '', 'fixed_lower': ''}
        if len(phase_list) > 1:
            for phase_i in range(len(phase_list)):
                for phase_j in range(len(phase_list)):
                    if phase_i != phase_j:
                        print(phase_i, phase_j)
                        root_image_path = '{}/{}'.format(root_path, joint)

                        source_file = '{}/{}.jpg'.format(root_image_path, phase_list[phase_i][:-4])
                        print(source_file)

                        row_list['joint_name'] = joint
                        row_list['moving'] = source_file
                        row_list['moving_upper'] = '{}/{}_mask_upper.bmp'.format(root_image_path,
                                                                                 phase_list[phase_i][:-4])
                        row_list['moving_lower'] = '{}/{}_mask_lower.bmp'.format(root_image_path,
                                                                                 phase_list[phase_i][:-4])

                        source_file = '{}/{}.jpg'.format(root_image_path, phase_list[phase_j][:-4])
                        row_list['fixed'] = source_file
                        row_list['fixed_upper'] = '{}/{}_mask_upper.bmp'.format(root_image_path,
                                                                                phase_list[phase_j][:-4])
                        row_list['fixed_lower'] = '{}/{}_mask_lower.bmp'.format(root_image_path,
                                                                                phase_list[phase_j][:-4])
                        json_list.append(row_list)

    save_path = '{}/DR_{}_{}'.format(file_path, cross, trainortest)
    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_list, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    root_path = ROOT_PATH + '/Dataset_cases'

    file_list = os.listdir(root_path)
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')
    for i in file_list:
        if i[0] == 'b':
            file_list.remove(i)


    def partition_array(array, cross_valuation_count=4):
        cases_list = []
        random.shuffle(array)
        for case in array:
            overlap_count = 0
            phase_list_org = os.listdir(root_path + '/' + case)
            phase_list_check = [file for file in phase_list_org if '.bmp' not in file]
            for image in phase_list_check:
                upper_mask = np.array(Image.open(root_path + '/' + case + '/' + image[:-4] + '_mask_upper.bmp'))
                lower_mask = np.array(Image.open(root_path + '/' + case + '/' + image[:-4] + '_mask_lower.bmp'))
                upper_mask[upper_mask != 0] = 1.0
                lower_mask[lower_mask != 0] = 1.0
                overlap_size = np.sum(cv2.bitwise_and(upper_mask, lower_mask))
                if overlap_size >= 20:
                    overlap_count += 1

            tmp = {'name': case, 'image_count': len(phase_list_check), 'overlap_count': overlap_count}
            cases_list.append(tmp)
        print(cases_list)

        # 计算数组长度的一半
        part_length = len(array) // cross_valuation_count

        # 将数组划分为多个部分

        part_list = []
        for i in range(cross_valuation_count):
            part_dict = cases_list[part_length * i:part_length * (i + 1)]
            if i == cross_valuation_count - 1:
                part_dict = cases_list[part_length * i: len(array)]
            part = []
            count = 0
            count_overlap = 0
            for i in part_dict:
                part.append(i['name'])
                count += i['image_count']
                count_overlap += i['overlap_count']
            print('images count: ', count, 'overlap image count:', count_overlap)
            part_list.append(part)

        return part_list


    while True:
        part_list = partition_array(file_list)
        user_input = input("True or False (y/n)")
        if user_input.lower() == 'y':
            break

    for part_idx in range(len(part_list)):
        part_list_tmp = copy.deepcopy(part_list)
        test_data_list = part_list_tmp.pop(part_idx)
        train_data_list = []
        for i in part_list_tmp:
            train_data_list += i

        print(len(test_data_list), len(train_data_list))

        cross = 'K{}'.format(part_idx + 1)
        makeup_data(train_data_list, file_path=ROOT_PATH, cross=cross, trainortest='train')
        makeup_data(test_data_list, file_path=ROOT_PATH, cross=cross, trainortest='test')
