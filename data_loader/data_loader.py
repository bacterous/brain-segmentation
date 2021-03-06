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

from data_loader.data_pre import *


def color_map():
    """
    color map:
    Background(dark), Cortical gray matter(gray), Basal ganglia(green),
    White matter(white), White matter lesions(Maroon),
    Cerebrospinal(olive), Ventricles(light blue), Cerebellum(red),
    Brain stem(teal), Infarction(purple), Other(cyan)
    """
    return np.asarray([[0, 0, 0], [128, 128, 128], [0, 255, 0],
                             [255, 255, 255], [128, 0, 0],
                             [128, 128, 0], [255, 255, 0], [255, 0, 0],
                             [0, 128, 128], [128, 0, 128], [0, 255, 255]]).astype(np.uint8)


class MRBrainS18(data.Dataset):
    def __init__(self, n_classes=11, root='data/training', val_num=5, is_val=False, is_transform=False, is_flip=False,
                 is_rotate=False, n_angles=5, is_crop=False, is_histeq=False, n_stack=5):
        self.root = root
        self.is_val = is_val
        self.n_classes = n_classes
        self.angles = np.zeros(1)

        self.color = color_map()
        self.label_test = [0, 2, 2, 3, 3, 1, 1, 0, 0]
        self.train_val_split(root, val_num)


        if is_val == False:
            print('training...')
            T1 = [to_uint8(read_vol(path)) for path in self.train_T1_path]
            IR = [IR_to_unit8(read_vol(path)) for path in self.train_IR_path]
            T2 = [to_uint8(read_vol(path)) for path in self.train_T2_path]
            label = [read_vol(path) for path in self.train_label_path]

            if is_flip:
                print('    flipping...')
                T1 = flip(T1)
                IR = flip(IR)
                T2 = flip(T2)
                label = flip(label)

            if is_histeq:
                print('    histogram equalizing...')
                T1 = [equal_hist(vol) for vol in T1]

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


    def train_val_split(self, root, val_num):
        # data paths
        names = os.listdir(root)
        T1_path = [root + name + '/pre/reg_T1.nii.gz' for name in names]
        IR_path = [root + name + '/pre/IR.nii.gz' for name in names]
        T2_path = [root + name + '/pre/FLAIR.nii.gz' for name in names]
        label_path = [root + name + '/segm.nii.gz' for name in names]

        # val data
        self.val_T1_path = T1_path[val_num - 1]
        self.val_IR_path = IR_path[val_num - 1]
        self.val_T2_path = T2_path[val_num - 1]
        self.val_label_path = label_path[val_num - 1]

        # train data
        self.train_T1_path = [item for item in T1_path if item not in self.val_T1_path]
        self.train_IR_path = [item for item in IR_path if item not in self.val_IR_path]
        self.train_T2_path = [item for item in T2_path if item not in self.val_T2_path]
        self.train_label_path = [item for item in label_path if item not in self.val_label_path]





