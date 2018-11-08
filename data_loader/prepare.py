from data_loader.pre_process import *

class Prepare(object):
    def __init__(self, root='data/training', save_root='data/train', val=5, parts=(6,2,2), patch_size=(8, 128, 128)):
        self.root = root
        self.save_root = save_root
        self.val = val
        self.parts = parts
        self.patch_size = patch_size

    def main(self):
        train_path, val_path = train_val_split(self.root, self.val)
        print('reading data ...')
        train_data, train_label = data_label_split(train_path) # (6, 3, 8, 240, 240), (6, 1, 8, 240, 240)
        print('histogram equalize ...')
        train_data = [series(patient, emphasize_edge) for patient in train_data]
        train_data = [series(patient, adaptive_hist) for patient in train_data]
        print('threshold based crop ...')
        train = np.concatenate((np.array(train_data), np.array(train_label)), 1) # (6, 4, 8, 240, 240)
        train = series(array_data(train), threshold_based_crop)
        print('partition & resize ...')
        train_patches = [series(patient, partition, self.parts) for patient in train]
        train_patches = series(train_patches, array_data)
        train_patches = [series(patient, series, resize, self.patch_size[::-1]) for patient in train_patches]   # (6, 4, 24, 8, 128, 128)

        train_data = series(train_patches, extract_label)
        train_label = series(train_patches, extract_label, False)
        print('intensity augmenting ...')
        train_data = [series(patient, series, augment_images_intensity) for patient in train_data]
        train_data = np.array([series(patient, data_array) for patient in train_data])  # (6, 3, 24, 3, 8, 128, 128)
        train_data = train_data.transpose((0, 1, 3, 2, 4, 5, 6)).reshape(6, 24, -1, 8, 128, 128)    # (6, 9, 24, 8, 128, 128)
        train_label = series(train_label, data_array).repeat(train_data.shape[1], 1) # (6, 9, 24, 8, 128, 128)

        print('generating file ...')
        for index in range(len(train_data)):
            file = {}
            file['data'] = train_data[index]
            file['label'] = train_label[index]
            npz_save(os.path.join(self.save_root, str(index)), file)



if __name__=='__main__':
    pre = Prepare()
    pre.main()