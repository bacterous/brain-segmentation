import os
import numpy as np
import cv2 as cv
import SimpleITK as sitk


def data_label_split(path):
    data = data_array([series(data_path, load_data) for data_path in path[:, :-1]])
    label = data_array(series(path[:, -1], load_segm))
    return data, label


def train_val_split(root, val_num):
    # data paths
    names = os.listdir(root)
    T1_path = [root + name + '/pre/reg_T1.nii.gz' for name in names]
    IR_path = [root + name + '/pre/IR.nii.gz' for name in names]
    T2_path = [root + name + '/pre/FLAIR.nii.gz' for name in names]
    label_path = [root + name + '/segm.nii.gz' for name in names]
    path = np.array([T1_path, IR_path, T2_path, label_path]).T

    val_path = path[val_num - 1].reshape(-1, 4)
    train_path = path[path != path[val_num - 1]].reshape(-1, 4)
    return np.array(train_path), np.array(val_path)


def extract_label(images, flag=True):
    data = images[:-1]
    label = images[-1]
    if flag:
        return data
    else:
        return [label]



def series(sequence, method=None, *args):
    """
    each object has a series of data, so organize them together
    :param sequence: sequence of data_path, images or patches
    :param method: load, flip, rotate, crop, histogram equalize, threshold
    :return: 
    """
    series_data = []
    for item in sequence:
        series_data.append(method(item, *args))

    return series_data


def data_array(data):
    """
    get image array from data
    """
    return [series(item, sitk.GetArrayFromImage) for item in data]


def array_data(array):
    """
    get image data from array
    """
    return [series(item, sitk.GetImageFromArray) for item in array]


def load_data(data_path):
    """
    load image
    """
    data = sitk.ReadImage(data_path)
    return sitk.Cast(sitk.RescaleIntensity(data), sitk.sitkUInt8)

def load_segm(seg_path):
    return [sitk.ReadImage(seg_path)]

def get_meta(data):
    """
    read meta data dictionary
    """
    meta_data = {}
    for k in data.GetMetaDataKeys():
        meta_data[k] = data.GetMetaData(k)

    return meta_data


def flip(volume, axis=1):
    """
    flipped horizontally, vertically or both
    :param volume: volume array, n_slices
    :param axis: horizontally: 1, vertically: 0 or both: -1
    :return: flipped data
    """
    return np.array([cv.flip(slice_, axis) for slice_ in volume])


def equalize_hist(volume):
    """
    histogram equalization on each slice
    :param volume: volume array
    :return: hist equalized volume array
    """
    return np.array([cv.equalizeHist(slice_) for slice_ in volume])


def adaptive_hist(volume, limit=2.0, grid=8):
    """
    adaptive histogram equalization on each slice
    :param volume: volume array
    :param limit: clip limit
    :param grid: tile size
    :return: hist equalized volume array
    """
    clahe = cv.createCLAHE(limit, (grid, grid))
    return np.array([clahe.apply(slice_) for slice_ in volume])


def emphasize_edge(volume):
    """
    emphasize edges using Sobel
    :param volume: volume array
    :return: processed array
    """
    for i in range(volume.shape[0]):
        cur_slice = volume[i, :, :]
        
        # in the case of 8-bit input images it will result in truncated derivatives, 8b -> CV_16S(16b)
        sob_x = cv.Sobel(cur_slice, cv.CV_16S, 1, 0)    # edges on axis X
        sob_y = cv.Sobel(cur_slice, cv.CV_16S, 0, 1)    # edges on axis Y
        
        # 16-bit -> 8-bit connvertScaleAbs()
        sob = cv.addWeighted(cv.convertScaleAbs(sob_x), 0.5,
                             cv.convertScaleAbs(sob_y), 0.5, 0)
        
        volume[i, :, :] = cur_slice + 0.5 * sob
        return volume
    

def rotate(patches, angle, interp):
    """
    rotated volume
    :param patches: patch list
    :param angle: rotate angle
    :param interp: interp method
    :return: rotated volume
    """
    for i, patch in enumerate(patches):
        rows, cols = patch.shape[0:2]
        M = cv.getRotationMatrix2D(((cols-1) / 2.0, (rows-1) / 2.0), angle, 1)
        patches[i] = cv.warpAffine(patch, M, (cols, rows), flags=interp)

    return patches


def resize(image, new_size):
    """
    resize image to new_size
    :param image:
    :param new_size:
    :return:
    """
    reference_image = sitk.Image(new_size, image.GetPixelIDValue())
    reference_image.SetOrigin(image.GetOrigin())
    reference_image.SetDirection(image.GetDirection())
    reference_image.SetSpacing([sz * spc / nsz for nsz, sz, spc in zip(new_size, image.GetSize(), image.GetSpacing())])
    return sitk.Resample(sitk.SmoothingRecursiveGaussian(image, 2.0), reference_image)


def threshold_bbox(image):
    """
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. And get the foreground's axis aligned bounding box
    :param image: An image where the anatomy and background intensities form a bi-modal distribution
    :return: Foreground's axis aligned bounding box.
    """
    inside_value = 0
    outside_value = 255

    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(sitk.OtsuThreshold(image, inside_value, outside_value))
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)

    return bounding_box


def threshold_based_crop(images):
    """
    Crop the image using the foreground's axis aligned bounding box.
    :param images: Images in different mode for one patient
    :return:  Cropped image based on foreground's axis aligned bounding box.
    """
    bounding_boxes = threshold_bbox(images[0])
    start_point = bounding_boxes[:int(len(bounding_boxes)/2)]
    box_size = bounding_boxes[int(len(bounding_boxes)/2):]

    return series(images, sitk.RegionOfInterest, box_size, start_point)


def augment_images_intensity(image):
    """
    Generate intensity modified images from the originals.
    :param image: The images which we whose intensities we modify.
    :return: augmented images
    """
    filter_list = []

    # Smoothing filters
    # filter_list.append(sitk.BilateralImageFilter())
    # filter_list[-1].SetDomainSigma(4.0)
    # filter_list[-1].SetRangeSigma(8.0)

    # Filter control via SetMean, SetStandardDeviation.
    filter_list.append(sitk.AdditiveGaussianNoiseImageFilter())

    # Filter control via SetScale
    filter_list.append(sitk.ShotNoiseImageFilter())

    aug_image = [f.Execute(image) for f in filter_list]

    return aug_image


def partition(image, n_parts):
    """
    evenly divided
    :param image: array, (z, y, z)
    :param n_parts: number of parts in each axis, (z_parts, y_parts, x_parts)
    :return: images after being partitioned
    """
    source_shape = np.shape(np.squeeze(image))
    increase = np.array(source_shape) // np.array(n_parts)

    images = []
    for z_i in range(n_parts[0]):
        z_start = z_i * increase[0]
        z_end = z_start + increase[0]

        for y_i in range(n_parts[1]):
            y_start = y_i * increase[1]
            y_end = y_start + increase[1]

            for x_i in range(n_parts[2]):
                x_start = x_i * increase[2]
                x_end = x_start + increase[2]

                part = np.squeeze(image)[z_start:z_end, y_start:y_end, x_start:x_end]
                images.append(part)

    return images


def merge(images, n_parts):
    """
    merge result from each patches
    :param images: array, (z_parts * y_parts * x_parts, )
    :param n_parts: array, (z_parts, y_parts, x_parts)
    :return:
    """
    return images.reshape(tuple(n_parts))


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


def npz_save(name, obj):
    keys = list(obj.keys())
    values = list(obj.values())
    np.savez(name + ".npz", keys=keys, values=values)

def npz_save_compressed(name, obj):
    keys = list(obj.keys())
    values = list(obj.values())
    np.savez_compressed(name+"_compressed.npz", keys=keys, values=values)

def npz_load(filename):
    npzfile = np.load(filename+".npz")
    keys = npzfile["keys"]
    values = npzfile["values"]
    return dict(zip(keys, values))

def npz_load_compressed(filename):
    npzfile = np.load(filename+"_compressed.npz")
    keys = npzfile["keys"]
    values = npzfile["values"]
    return dict(zip(keys, values))


if __name__=='__main__':
    data_path = "l"
    load_data(data_path)
