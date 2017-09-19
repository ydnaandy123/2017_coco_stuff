import os
from glob import glob
import numpy as np
import scipy.io as sio
import scipy.ndimage
import scipy.misc


class MSCOCOParser:
    def __init__(self, dataset_dir, target_height=256, target_width=256):
        self.target_height, self.target_width = target_height, target_width
        self.dataset_dir = dataset_dir

        self.images_train_dir = self.dataset_dir + '/images/train2017'
        self.images_val_dir = self.dataset_dir + '/images/val2017'
        self.annotations_train_dir = self.dataset_dir + '/annotations/stuff_train2017_labelmaps'
        self.annotations_val_dir = self.dataset_dir + '/annotations/stuff_val2017_labelmaps'

        self.images_val_paths, self.annotations_val_paths, self.val_paths = [], [], []
        self.images_train_paths, self.annotations_train_paths, self.train_paths = [], [], []

    def load_val_paths(self):
        self.annotations_val_paths = sorted(glob(os.path.join(self.annotations_val_dir, "*.png")))
        self.images_val_paths = [annotations_val_path.replace(
            self.annotations_val_dir, self.images_val_dir).replace(
            '.png', '.jpg') for annotations_val_path in self.annotations_val_paths]
        self.val_paths = list(zip(self.images_val_paths, self.annotations_val_paths))
        return self

    def load_train_paths(self):
        self.annotations_train_paths = sorted(glob(os.path.join(self.annotations_train_dir, "*.png")))
        self.images_train_paths = [annotations_train_path.replace(
            self.annotations_train_dir, self.images_train_dir).replace(
            '.png', '.jpg') for annotations_train_path in self.annotations_train_paths]
        self.train_paths = list(zip(self.images_train_paths, self.annotations_train_paths))
        return self

    def load_train_datum_batch(self, start, end):
        batch_len = end - start
        train_paths_batch = self.train_paths[start:end]
        x_batch, y_batch = [], []
        for idx, train_path in enumerate(train_paths_batch):
            x = scipy.misc.imread(train_path[0])
            y = scipy.misc.imread(train_path[1])
            x = scipy.misc.imresize(x, (self.target_height, self.target_width))
            y = scipy.misc.imresize(y, (self.target_height, self.target_width), interp='nearest')
            if idx >= batch_len // 2:
                x = np.fliplr(x)
                y = np.fliplr(y)
            if len(x.shape) < 3:
                x = np.dstack((x, x, x))
            x_batch.append(x)
            y_batch.append(y)
        return x_batch, y_batch

    def load_train_datum_batch_aug(self, start, end):
        print('loading training datum batch...')
        mat_train_paths_batch = self.mat_train_paths[start:end]
        x_batch, y_batch = [], []
        for idx, mat_train_path in enumerate(mat_train_paths_batch):
            mat_contents = sio.loadmat(mat_train_path)
            x, y = mat_contents['sample'][0][0]['RGBSD'], mat_contents['sample'][0][0]['GT']
            x, y = self.data_augmentation(x=x, y=y)
            x_batch.append(x)
            y_batch.append(y)
        return x_batch, y_batch

    def load_val_datum_batch(self, start, end):
        valid_paths_batch = self.val_paths[start:end]
        x_batch, y_batch = [], []
        for idx, valid_path in enumerate(valid_paths_batch):
            x = scipy.misc.imread(valid_path[0])
            y = scipy.misc.imread(valid_path[1])
            x_batch.append(x)
            y_batch.append(y)
        return x_batch, y_batch

    def data_augmentation(self, x, y):
        coin = np.random.randint(0, 2)
        """
        Flip
        """
        if coin == 1:
            x = np.fliplr(x)
            y = np.fliplr(y)

        """
        Color
        """
        x = x.astype(np.float32)
        x[:, :, 0] *= np.random.uniform(0.6, 1.4)
        x[:, :, 1] *= np.random.uniform(0.6, 1.4)
        x[:, :, 2] *= np.random.uniform(0.6, 1.4)
        x[:, :, :3] = np.minimum(x[:, :, :3], 255)
        x = x.astype(np.uint8)

        """
        Rotation
        """
        angle = np.random.uniform(-30.0, 30.0)
        x = scipy.ndimage.interpolation.rotate(input=x, angle=angle, axes=(1, 0))
        y = scipy.ndimage.interpolation.rotate(input=y, angle=angle, axes=(1, 0))

        '''
        """
        Zoom
        """
        zoom = np.random.uniform(1.2, 1.2)
        x = scipy.ndimage.interpolation.zoom(input=x, zoom=zoom)
        y = scipy.ndimage.interpolation.zoom(input=y, zoom=zoom)
        '''

        """
        Crop
        """
        height, width, channel = np.shape(x)
        off_h, off_w = 0, 0
        padding = height - self.target_height
        if padding != 0:
            off_h = np.random.randint(0, padding)
            off_w = np.random.randint(0, padding)
        x = x[off_h:off_h+self.target_height, off_w:off_w+self.target_width, :]
        y = y[off_h:off_h+self.target_height, off_w:off_w+self.target_width]

        return x, y


class ELECParser:
    def __init__(self, dataset_dir, target_height=256, target_width=256):
        self.target_height, self.target_width = target_height, target_width
        self.dataset_dir = dataset_dir

        self.images_train_dir = self.dataset_dir + '/images/training_c'
        self.images_val_dir = self.dataset_dir + '/images/training_c'
        self.annotations_train_dir = self.dataset_dir + '/annotations/training_c'
        self.annotations_val_dir = self.dataset_dir + '/annotations/training_c'

        self.images_val_paths, self.annotations_val_paths, self.val_paths = [], [], []
        self.images_train_paths, self.annotations_train_paths, self.train_paths = [], [], []

    def load_val_paths(self):
        self.annotations_val_paths = sorted(glob(os.path.join(self.annotations_val_dir, "*.png")))
        self.images_val_paths = [annotations_val_path.replace(
            self.annotations_val_dir, self.images_val_dir) for annotations_val_path in self.annotations_val_paths]
        self.val_paths = list(zip(self.images_val_paths, self.annotations_val_paths))
        return self

    def load_train_paths(self):
        self.annotations_train_paths = sorted(glob(os.path.join(self.annotations_train_dir, "*.png")))
        self.images_train_paths = [annotations_train_path.replace(
            self.annotations_train_dir, self.images_train_dir) for annotations_train_path in self.annotations_train_paths]
        self.train_paths = list(zip(self.images_train_paths, self.annotations_train_paths))
        return self

    def load_train_datum_batch(self, start, end):
        batch_len = end - start
        train_paths_batch = self.train_paths[start:end]
        x_batch, y_batch = [], []
        for idx, train_path in enumerate(train_paths_batch):
            x = scipy.misc.imread(train_path[0])[:, :, :3]
            y = scipy.misc.imread(train_path[1])[:, :, 0]
            x = scipy.misc.imresize(x, (self.target_height, self.target_width))
            y = scipy.misc.imresize(y, (self.target_height, self.target_width), interp='nearest')
            if idx >= batch_len // 2:
                x = np.fliplr(x)
                y = np.fliplr(y)
            x_batch.append(x)
            y_batch.append(y)
        return x_batch, y_batch

    def load_train_datum_batch_aug(self, start, end):
        train_paths_batch = self.train_paths[start:end]
        x_batch, y_batch = [], []
        for idx, train_path in enumerate(train_paths_batch):
            x = scipy.misc.imread(train_path[0])[:, :, :3]
            y = scipy.misc.imread(train_path[1])[:, :, 0]
            x = scipy.misc.imresize(x, (self.target_height, self.target_width))
            y = scipy.misc.imresize(y, (self.target_height, self.target_width), interp='nearest')
            # if idx >= batch_len // 2:
            #     x = np.fliplr(x)
            #     y = np.fliplr(y)
            x, y = self.data_augmentation(x=x, y=y)
            x_batch.append(x)
            y_batch.append(y)
        return x_batch, y_batch

    def load_val_datum_batch(self, start, end):
        valid_paths_batch = self.val_paths[start:end]
        x_batch, y_batch = [], []
        for idx, valid_path in enumerate(valid_paths_batch):
            x = scipy.misc.imread(valid_path[0])
            y = scipy.misc.imread(valid_path[1])
            x_batch.append(x)
            y_batch.append(y)
        return x_batch, y_batch

    def data_augmentation(self, x, y):
        coin = np.random.randint(0, 2)
        """
        Flip
        """
        if coin == 1:
            x = np.fliplr(x)
            y = np.fliplr(y)

        """
        Color
        """
        x = x.astype(np.float32)
        x[:, :, 0] *= np.random.uniform(0.6, 1.4)
        x[:, :, 1] *= np.random.uniform(0.6, 1.4)
        x[:, :, 2] *= np.random.uniform(0.6, 1.4)
        x[:, :, :3] = np.minimum(x[:, :, :3], 255)
        x = x.astype(np.uint8)

        """
        Rotation
        """
        angle = np.random.uniform(-30.0, 30.0)
        x = scipy.ndimage.interpolation.rotate(input=x, angle=angle, axes=(1, 0))
        y = scipy.ndimage.interpolation.rotate(input=y, angle=angle, axes=(1, 0))

        '''
        """
        Zoom
        """
        zoom = np.random.uniform(1.2, 1.2)
        x = scipy.ndimage.interpolation.zoom(input=x, zoom=zoom)
        y = scipy.ndimage.interpolation.zoom(input=y, zoom=zoom)
        '''

        """
        Crop
        """
        height, width, channel = np.shape(x)
        off_h, off_w = 0, 0
        padding = height - self.target_height
        if padding != 0:
            off_h = np.random.randint(0, padding)
            off_w = np.random.randint(0, padding)
        x = x[off_h:off_h+self.target_height, off_w:off_w+self.target_width, :]
        y = y[off_h:off_h+self.target_height, off_w:off_w+self.target_width]

        return x, y