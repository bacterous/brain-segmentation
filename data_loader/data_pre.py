import random
import math
import numpy as np
import cv2 as cv
import nibabel as nib
import SimpleITK as sitk


def load_data(data_path):
    """
    load image
    """
    data = sitk.ReadImage(data_path)
    return sitk.Cast(sitk.RescaleIntensity(data), sitk.sitkUInt8)


def load_array(data_path):
    """
    read image array from data
    """
    return sitk.GetArrayFromImage(load_data(data_path))


def load_meta(data_path):
    """
    read meta data dictionary
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(data_path)

    reader.Execute()
    reader.ReadImageInformation()

    meta_data = {}
    for k in reader.GetMetaDataKeys():
        meta_data[k] = reader.GetMetaData(k)

    return meta_data


def series(data_paths, method=None):
    """
    each object has a series of data, so organize them together

    """
    series_data = []
    for data_path in data_paths:
        series_data.append(method(data_path))

    return series_data



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
    vol[vol < 0] = 0
    return ((vol - vol.min()) * 255.0 / vol.max()).astype(np.uint8)


def IR_to_unit8(vol):
    """
    IR comprise, 0 where <800
    :param vol: volume array
    :return: comprised data
    """
    vol = vol.astype(np.float)
    vol[vol < 0] = 0
    return ((vol - 800) * 255 / vol.max()).astype(np.uint8)


def flip(data, axis=1):
    """
    flipped horizontally, vertically or both
    :param data: raw data n_patients * n_slices
    :param axis: horizontally: 1, vertically: 0 or both: -1
    :return: flipped data
    """
    for i in range(len(data)):
        data.append(np.array([cv.flip(slice_, axis) for slice_ in data[i]]))

    return data


def equal_hist(vol):
    """
    histogram equalization
    :param vol: volume array
    :return: hist equalized volume array
    """
    for slice_index in range(vol.shape[2]):
        vol[:, :, slice_index] = cv.equalizeHist(vol[:, :, slice_index])

    return vol


def clahe_hist(vol):
    """
    adaptive histogram equalization
    :param vol: volume array
    :return: hist equalized volume array
    """
    for slice_index in range(vol.shape[2]):
        vol[:, :, slice_index] = cv.equalizeHist(vol[:, :, slice_index])

    return vol


def emph_edge(vol):
    """
    emphasize edges
    :param vol: volume array
    :return: processed array
    """
    for slice_index in range(vol.shape[2]):
        cur_slice = vol[:, :, slice_index]
        # in the case of 8-bit input images it will result in truncated derivatives, 8b -> CV_16S(16b)
        sob_x = cv.Sobel(cur_slice, cv.CV_16S, 1, 0)
        sob_y = cv.Sobel(cur_slice, cv.CV_16S, 0, 1)
        # 16-bit -> 8-bit connvertScaleAbs()
        sob = cv.addWeighted(cv.convertScaleAbs(sob_x), 0.5,
                             cv.convertScaleAbs(sob_y), 0.5, 0)
        vol[:, :, slice_index] = cur_slice + 0.5*sob
        return vol


def get_stack_index(slice_index, n_stack, slice_num):
    """
    calculate index around the slice which waits for stacking
    :param slice_index: center slice
    :param n_stack: stack n slice into one patch
    :param slice_num: counts of all slices
    :return: list, index around center slice
    """
    assert n_stack, 'stack numbers must be odd!'
    query_list = [0] * n_stack
    for stack_index in range(n_stack):
        query_list[stack_index] = (slice_index + (stack_index - int(n_stack/2))) % slice_num

    return  query_list


def get_stacked(vol, n_stack):
    """
    stack n_stack slices into a patch
    :param vol: (h, w, n_slice)
    :param n_stack: n slice per patch
    :return: stacked vol
    """
    stack_list = []
    stacked_slice = np.zeros((vol.shape[0], vol.shape[1], n_stack), np.uint8)
    for slice_index in range(vol.shape[2]):
        query_list = get_stack_index(slice_index, n_stack, vol.shape[2])
        for index, content in enumerate(query_list):
            stacked_slice[:, :, index] = vol[:, :, content].transpose()

        stack_list.append(stacked_slice.copy())

    return stack_list


def rotate(stack_list, angle, interp):
    """
    rotate volume
    :param stack_list: patch list
    :param angle: rotate angle
    :param interp: interp method
    :return: after rotated volume
    """
    for index, stacked in enumerate(stack_list):
        rows, cols = stacked.shape[0:2]
        M = cv.getRotationMatrix2D(((cols-1) / 2.0, (rows-1) / 2.0), angle, 1)
        stack_list[index] = cv.warpAffine(stacked, M, (cols, rows), flags=interp)

    return  stack_list


def calc_crop_region(stack_T1, threshold, pix):
    """

    :param stack_T1:
    :param threshold:
    :param pix:
    :return:
    """
    crop_region = []
    for index, stacked in enumerate(stack_T1):
        _, threshold_img = cv.threshold(stacked[:, :, int(stacked.shape[2] / 2)].copy(), threshold, 255, cv.THRESH_TOZERO)
        pix_index = np.where(threshold_img > 0)
        if not pix_index[0].size == 0:
            y_min, y_max = min(pix_index[0]), max(pix_index[0])
            x_min, x_max = min(pix_index[1]), max(pix_index[1])
        else:
            y_min, y_max = pix, pix
            x_min, x_max = pix, pix

        y_min = (y_min <= pix) and (0) or (y_min)
        y_max = (y_max >= stacked.shape[0]-1-pix) and (stacked.shape[0] - 1) or (y_max)
        x_min = (x_min <= pix) and (0) or (x_min)
        x_max = (x_max >= stacked.shape[1]-1-pix) and (stacked.shape[1] - 1) or (x_max)
        crop_region.append([y_min, y_max, x_min, x_max])

    return crop_region


def calc_max_region_list(region_list, stack_num):
    """

    :param region_list:
    :param stack_num:
    :return:
    """
    max_region_list = []
    for region_index in range(len(region_list)):
        y_min_list, y_max_list = [], []
        x_min_list, x_max_list = [], []
        for stack_index in range(stack_num):
            query_list = get_stack_index(region_index, stack_num, len(region_list))
            region = region_list[query_list[stack_index]]
            y_min_list.append(region[0])
            y_max_list.append(region[1])
            x_min_list.append(region[2])
            x_max_list.append(region[3])

        max_region_list.append([min(y_min_list), max(y_max_list), min(x_min_list), max(x_max_list)])

    return max_region_list


def calc_ceil_pad(x, devider):
    return math.ceil(x / float(devider)) * devider


def crop(stack_list, region_list):
    """

    :param stack_list:
    :param region_list:
    :return:
    """
    cropped_list = []
    for index, stacked in enumerate(stack_list):
        y_min, y_max, x_min, x_max = region_list[index]
        cropped = np.zeros((calc_ceil_pad(y_max-y_min, 16), calc_ceil_pad(x_max-x_min, 16), stacked.shape[2]), np.uint8)
        cropped[0:y_max-y_min, 0:x_max-x_min, :] = stacked[y_min:y_max, x_min:x_max, :]
        cropped_list.append(cropped.copy())

    return cropped_list


def get_edge(stack_list, kernel_size=(3, 3), sigmaX=0):
    """

    :param stack_list:
    :param kernel_size:
    :param sigmaX:
    :return:
    """
    edge_list = []
    for stacked in stack_list:
         edges = np.zeros((stacked.shape[0], stacked.shape[1], stacked.shape[2]), np.uint8)
         for slice_index in range(stacked.shape[2]):
             edges[:, :, slice_index] = cv.Canny(stacked[:, :, slice_index], 1, 1)
             edges[:, :, slice_index] = cv.GaussianBlur(edges[:, :, slice_index], kernel_size, sigmaX)
             cv.imshow('edges', edges[:, :, slice_index])
             cv.waitKey(0)

         edge_list.append(edges)

    return edge_list


if __name__=='__main__':
    T1_path='data/training/1/orig/reg_T1.nii.gz'
    pre_T1_path = 'data/training/1/pre/reg_T1.nii.gz'
    seg_path = 'data/training/1/pre/FLAIR.nii.gz'
    vol=to_uint8(read_vol(T1_path))
    # pre_vol = to_uint8(read_vol(pre_T1_path))
    seg = to_uint8(read_vol(seg_path))
    # print(seg.shape)
    # print(vol.shape)
    # histeqed = equal_hist(vol)
    # pre_hist = equal_hist(pre_vol)
    seg_eq = equal_hist(seg)
    cv.imshow('seg', seg[:, :, 24])
    cv.imshow('seg_eq', seg_eq[:, :, 24])
    cv.waitKey(0)
    # for i in range(vol.shape[2]):
    #     # cv.imshow('orig', vol[:,:,i])
    #     # cv.imshow('pre', pre_hist[:, :, i])
    #     cv.imshow('seg', seg[:, :, i])
    #     cv.imshow('seg_eq', seg_eq[:, :, i])
    #     cv.waitKey(0)
    # print('vol[100,100,20]= ', vol[100,100,20])
    # histeqed=equal_hist(vol)
    # cv.imshow('hist', histeqed[:, :, 20])
    # cv.waitKey(0)
    # print('vol[100,100,20]= ', vol[100,100,20])
    # print('query list: ', get_stack_index(2,5,histeqed.shape[2]))
    # stack_list=get_stacked(histeqed,5)
    # print(np.array(stack_list).shape)
    # print(len(stack_list))
    # print(stack_list[0].shape)
    # angle=random.uniform(-15,15)
    # print('angle= ', angle)
    # rotated=rotate(stack_list,angle, cv.INTER_LINEAR)
    # print(len(rotated))
    # region=calc_crop_region(rotated,50,5)
    # max_region=calc_max_region_list(region,5)
    # print(region)
    # print(max_region)
    # cropped=crop(rotated,max_region)
    # for i in range(48):
    #     print(cropped[i].shape)
    # get_edge(stack_list)
