import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import datetime
import logging
import torch.nn as nn
import torch
from itertools import islice

LOG_ROOT_PATH = './log/'

def monitor_view():
    log_file_path = LOG_ROOT_PATH + 'monitor.log'
    epochs = 0
    line_counts = 0
    log_data = {}
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            epochs += 1
            if log_data == {}:
                line_data = line.strip().split(',')
                for i in range(3, len(line_data)):
                    line_counts += 1
                    key, value = line_data[i].split(':')
                    log_data.setdefault(key[1:], [])
                # print(log_data)
            line_data = line.strip().split(',')
            for i in range(3, len(line_data)):
                key, value = line_data[i].split(':')
                log_data[key[1:]].append(float(value))

    epoch = range(0, epochs)
    fig = plt.figure(figsize=(5, 4 * line_counts))
    index = 1
    for key, value in log_data.items():
        plt.subplot(line_counts, 1, index)
        plt.plot(epoch, log_data[key], '.-')
        
        plt.title('{}'.format(key))
        plt.xlabel('Epoch')
        plt.grid(True)
        index += 1
    plt.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.1, wspace=0.3)
    plt.show()

class Monitor():
    def __init__(self, epochs, device, train_loss_name_list, val_loss_name_list, lr_name_list, train_dataset, val_dataset):
        self.epochs = epochs
        self.epoch_count = 0
        self.step_count = 0

        self.train_dataset_length = len(train_dataset)
        self.val_dataset_length = len(val_dataset)

        self.loss_dict = {}
        self.lr_dict = {}
        self.val_image_list = []

        self.switch = 'train'

        for loss_name in train_loss_name_list:
            self.loss_dict.setdefault(loss_name, Loss_monitor(loss_name=loss_name, type='train'))
        for loss_name in val_loss_name_list:
            self.loss_dict.setdefault(loss_name, Loss_monitor(loss_name=loss_name, type='val'))
        for lr_name in lr_name_list:
            self.lr_dict.setdefault(lr_name, LR_monitor(lr_name=lr_name))


        log_path = LOG_ROOT_PATH + 'monitor.log'
        with open(log_path, 'w'):
            pass
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s, %(message)s')
        file_handler.setFormatter(formatter)

        self.logger = logging.getLogger('Logger')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

        print('-' * 50)
        print('Device: {}'.format(device))
        current_time = datetime.datetime.now()
        print('Start Time:', current_time)
        print("-" * 50)

    def __str__(self):
        str_line = 'epochs: {}'.format(self.epochs) + '\n train - val: {} - {}'.format(self.train_dataset_length, self.val_dataset_length)
        for key, value in self.loss_dict.items():
            str_line = str_line + '\n{}: {}'.format(key, self.loss_dict[key].type)

        for key, value in self.lr_dict.items():
            str_line = str_line + '\n{}'.format(key)
        return str_line

    def train_start(self, optimizer_list=[]):
        self.switch = 'train'
        self.step_count = 0
        print('-' * 43 + ' Train Model ' + '-' * 44)
        print('Epoch: {} / {}'.format(self.epoch_count + 1, self.epochs))
        index = 0
        for key, value in self.lr_dict.items():
            self.lr_dict[key].input_lr(optimizer_list[index])
            index += 1

    def val_start(self):
        self.switch = 'val'
        self.step_count = 0
        print('-' * 43 + ' Valid Model ' + '-' * 44)
        print('Epoch: {} / {}'.format(self.epoch_count + 1, self.epochs))

    def set_output_image(self, number=1, image_list=[]):
        interval = self.val_dataset_length // number
        if self.step_count % interval == 0:
            self.val_image_list.append(image_list)

    def show_val_result(self):
        for image_box in self.val_image_list:
            plt.subplot(2, 3, 1)
            plt.imshow(image_box[0].cpu().detach().numpy()[0][0], 'gray')
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(2, 3, 4)
            plt.imshow(image_box[2].cpu().detach().numpy()[0][0], 'gray')
            plt.title('Reconstructed Image')
            plt.axis('off')

            plt.subplot(2, 3, 2)
            plt.imshow(image_box[1].cpu().detach().numpy()[0][0], 'gray')
            plt.title('Pre Layer (Lower)')
            plt.axis('off')

            plt.subplot(2, 3, 3)
            plt.imshow(image_box[1].cpu().detach().numpy()[0][1], 'gray')
            plt.title('Pre Layer (Upper)')
            plt.axis('off')

            plt.subplot(2, 3, 5)
            plt.imshow(image_box[3].cpu().detach().numpy()[0][0], 'gray')
            plt.title('Pre Mask (Lower)')
            plt.axis('off')

            plt.subplot(2, 3, 6)
            plt.imshow(image_box[3].cpu().detach().numpy()[0][1], 'gray')
            plt.title('Pre Mask (Upper)')
            plt.axis('off')

            plt.show()
        
        self.val_image_list = []


    def set_loss(self, loss_list=[]):
        index = 0
        self.step_count += 1
        if self.switch == 'train':
            print_info = 'Epoch: {}, Step: {}/{}, '.format(self.epoch_count + 1, self.step_count, self.train_dataset_length)

        if self.switch == 'val':
            print_info = 'Epoch: {}, Step: {}/{}, '.format(self.epoch_count + 1, self.step_count, self.val_dataset_length)

        for key, value in self.loss_dict.items():
            if self.loss_dict[key].type == self.switch:
                print_info = print_info + '{}: {:.12f} '.format(key, loss_list[index])
                self.loss_dict[key].input_step_loss(loss_list[index])
                index += 1
        print(print_info)

    def epoch_summary(self):
        print('-' * 42 + ' Epoch Summary ' + '-' * 43)
        log_info = '{}, '.format(self.epoch_count + 1) + ', '.join(['{}: {:.6f}'.format(lr_key, self.lr_dict[lr_key].lr_list[self.epoch_count]) for index, lr_key in
                                    enumerate(self.lr_dict.keys())]) + ', '
        for key, value in self.loss_dict.items():
            if self.loss_dict[key].type == 'train':
                print_info = 'Train Loss --> '
                loss_mean, loss_var = self.loss_dict[key].epoch_loss_summary()
                print_info = print_info + '{}: {:.12f}({:.8f})'.format(key, loss_mean, loss_var)
                log_info = log_info + '{}-train: {:.12f}, '.format(key, loss_mean)
                print(print_info)
            if self.loss_dict[key].type == 'val':
                print_info = 'Valid Loss --> '
                loss_mean, loss_var = self.loss_dict[key].epoch_loss_summary()
                print_info = print_info + '{}: {:.12f}({:.8f})'.format(key, loss_mean, loss_var)
                log_info = log_info + '{}-val: {:.12f}'.format(key, loss_mean)
                print(print_info)
        print("=" * 100)
        self._set_log(log_info=log_info)
        self.epoch_count += 1

    def get_recent_best_loss(self, loss_name=''):
        return self.loss_dict[loss_name].get_recent_loss(self.epoch_count - 1)

    def _set_log(self, log_info):
        self.logger.info(log_info)



class Loss_monitor():
    """
        loss date_type: float
        loss list data_type: list
    """

    def __init__(self, loss_name='loss', type='tran'):
        self.loss_name = loss_name
        self.loss_list = []

        self.type = type

        self.step_loss_list = []
        self.log_path = LOG_ROOT_PATH

    def input_step_loss(self, loss):
        self.step_loss_list.append(loss.item())

    def epoch_loss_summary(self):
        loss_mean = np.mean(self.step_loss_list)
        self.loss_list.append(loss_mean)
        loss_var = np.var(self.step_loss_list)

        return loss_mean, loss_var

    def get_recent_loss(self, epoch):
        return self.loss_list[epoch]

    def __str__(self):
        return '{}: {}'.format(self.loss_name, self.loss_list)

class LR_monitor():
    def __init__(self, lr_name):
        self.lr_name = lr_name
        self.lr_list = []

    def input_lr(self, optimizer):
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        self.lr_list.append(lr)




