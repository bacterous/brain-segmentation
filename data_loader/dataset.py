import os
import math
import random

import cv2 as cv
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib as plt

import torch
import torchvision.transforms as transforms
from torch.utils import data

from data_loader.pre_process import *


class MRBrainS18(data.Dataset):
    def __init__(self, root='data/training', val=5, train=True, transform=None):
        self.root = root
        self.val = val
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        train_path, val_path = train_val_split(self.root, self.val)
        if self.train:
            patient, segmentation = load_data(train_path[index, :3]), load_segm(train_path[index, 3])
        else:
            patient, segmentation = load_data(val_path[index, :3]), load_segm(val_path[index, 3])



        self.train_val_split(root, val_num)

        if is_val == False:
            print('training...')
            data = [series(path, load_data) for path in self.train_path]

            data = data_array(data)
            data = [series(patient, emphasize_edge) for patient in data]
            data = [series(patient, adaptive_hist) for patient in data]

            data = array_data(data)
            data = series(data, threshold_based_crop)
            aug_data = [series(patient, augment_images_intensity) for patient in data]
            aug_data = np.array(series(aug_data, data_array))
            aug_data = aug_data.reshape(aug_data.shape[0], -1, aug_data.shape[-3], aug_data.shape[-2], aug_data.shape[-1])
            data = np.concatenate([data_array(data), aug_data], 1)
            parts = np.array([6,2,2])
            for i, patient in enumerate(data):
                patient_data = np.array(series(patient, partition, parts))






            data.append([series(patient, flip) for patient in data])





            print('    getting stacked...')
            T1_stacks = [get_stacked(vol, n_stack) for vol in T1]
            IR_stacks = [get_stacked(vol, n_stack) for vol in IR]
            T2_stacks = [get_stacked(vol, n_stack) for vol in T2]
            label_stacks = [get_stacked(vol, n_stack) for vol in label]

            if is_rotate:
                print('    rotating...')
                self.angles = np.append(self.angles, np.random.uniform(-15, 15, n_angles-1), 0)
                for angle in self.angles:
                    for i in range(len(T1_stacks)):
                        T1_stacks.append(rotate(T1_stacks[i], angle, interp=cv.INTER_CUBIC).copy())
                        IR_stacks.append(rotate(IR_stacks[i], angle, interp=cv.INTER_CUBIC).copy())
                        T2_stacks.append(rotate(T2_stacks[i], angle, interp=cv.INTER_CUBIC).copy())
                        label_stacks.append(rotate(label_stacks[i], angle, interp=cv.INTER_CUBIC).copy())

            if is_crop:
                print('    cropping...')
                regions = [calc_max_region_list(calc_crop_region(T1_stack, 50, 5), n_stack) for T1_stack in T1_stacks]
                T1_stacks = [crop(stack_list, regions[i]) for i, stack_list in enumerate(T1_stacks)]
                IR_stacks = [crop(stack_list, regions[i]) for i, stack_list in enumerate(IR_stacks)]
                T2_stacks = [crop(stack_list, regions[i]) for i, stack_list in enumerate(T2_stacks)]
                label_stacks = [crop(stack_list, regions[i]) for i, stack_list in enumerate(label_stacks)]

            # get means
            T1_mean = np.mean(T1_stacks)
            IR_mean = np.mean(IR_stacks)
            T2_mean = np.mean(T2_stacks)

            # get edges
            print('getting edges')
            edge_stacks = []
            for samples in label_stacks:
                edge_stacks.append(get_edge(samples))

            # transform
            if is_transform:
                print('    transforming...')
                T1_stacks, IR_stacks, T2_stacks, label_stacks, edge_stacks \
                    = self.transfrom(T1_stacks, IR_stacks, T2_stacks, label_stacks, edge_stacks)

        else:
            print('validating...')
            T1 = to_uint8(read_vol(self.val_T1_path))
            IR = to_uint8(read_vol(self.val_IR_path))
            T2 = to_uint8(read_vol(self.val_T2_path))
            label = to_uint8(read_vol(self.val_label_path))

            if is_histeq:
                print('    hist equalizing...')
                T1 = equal_hist(T1)

            print('get stacking...')
            T1_stacks = get_stacked(T1, n_stack)
            IR_stacks = get_stacked(IR, n_stack)
            T2_stacks = get_stacked(T2, n_stack)
            label_stacks = get_stacked(label, n_stack)

            if is_crop:
                print('    cropping...')
                regions = calc_max_region_list(calc_crop_region(T1_stacks, 50, 5), n_stack)
                T1_stacks = crop(T1_stacks, regions)
                IR_stacks = crop(IR_stacks, regions)
                T2_stacks = crop(T2_stacks, regions)
                label_stacks = crop(label_stacks, regions)

            # get edges
            print('getting edges')
            edge_stacks = get_edge(label_stacks)

            # transform
            if is_transform:
                print('    transforming')
                T1_stacks, IR_stacks, T2_stacks, label_stacks, edge_stacks = self.transfrom(
                    np.expand_dims(np.array(T1_stacks), 0),
                    np.expand_dims(np.array(IR_stacks), 0),
                    np.expand_dims(np.array(T2_stacks), 0),
                    np.expand_dims(np.array(label_stacks), 0),
                    np.expand_dims(np.array(edge_stacks), 0),
                )
        # data ready
        self.T1_stacks = T1_stacks
        self.IR_stacks = IR_stacks
        self.T2_stacks = T2_stacks
        self.lbl_stacks = label_stacks
        self.edge_stacks = edge_stacks

        def __len__(self):
            return self.is_val and 48 or 48*6*7*2


        def __getitem__(self, index):
            if self.is_val == False:
                set_index = range(len(self.T1_stacks))











    def transfrom(self, *inputs):
        outputs = np.array([])
        for input in inputs:
            np.append(outputs, (input.transpose(0, 1, 4, 2, 3).astype(np.float) - np.mean(input))/255.0 )

        return outputs







