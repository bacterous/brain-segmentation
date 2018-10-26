import os
import random
import math
import torch
import numpy as np
import nibabel as nib

def read_vol(vol_path):
    """
    read data
    :param vol_path: volume path
    :return: volume data in array
    """
    return nib.load(vol_path).get_data()

def to_uint8(vol):
    """
    comprise to (0, 255), 0 where < 0
    :param vol: volume array
    :return: comprised data
    """
    vol = vol.astype(np.float)
    vol[vol<0] = 0
    return ((vol - vol.min()) * 255.0 / vol.max()).astype(np.unit8)

def IR_to_unit8(vol):
    """
    IR comprise, 0 where <800
    :param vol: volume array
    :return: comprised data
    """
    vol = vol.astype(np.float)
    vol[vol<0] = 0
    return ((vol - 800) * 255 / vol.max()).astype(np.uint8)

def histeq(vol):
    for slice_index in range
