import os
import scipy.ndimage
import scipy.misc
import tensorflow as tf
import numpy as np
from glob import glob
from PIL import Image
from pycocotools import cocostuffhelper
from pycocotools.coco import COCO


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class MSCOCOParser:
    def __init__(self, dataset_dir, flags):
        self.image_height, self.image_width = flags.image_height, flags.image_width
        self.dataset_dir = dataset_dir

        self.images_train_dir = self.dataset_dir + '/images/train2017'
        self.images_val_dir = self.dataset_dir + '/images/val2017'
        self.images_test_dir = self.dataset_dir + '/images/test2017'
        self.annotations_train_dir = self.dataset_dir + '/annotations/stuff_train2017_pixelmaps'
        self.annotations_val_dir = self.dataset_dir + '/annotations/stuff_val2017_pixelmaps'
        self.annotations_test_dev_info = self.dataset_dir + '/annotations/image_info_test-dev2017.json'
        self.annotations_test_info = self.dataset_dir + '/annotations/image_info_test2017.json'
        self.TFRecord_dir = './dataset/coco_stuff_TFRecord'

        self.images_val_paths, self.annotations_val_paths, self.val_paths = [], [], []
        self.images_train_paths, self.annotations_train_paths, self.train_paths = [], [], []

        cmap = np.array(cocostuffhelper.getCMap())
        cmap = (cmap * 255).astype(int)
        padding = np.zeros((256 - cmap.shape[0], 3), np.int8)
        cmap = np.vstack((cmap, padding))
        cmap = cmap.reshape((-1))
        assert len(cmap) == 768, 'Error: Color map must have exactly 256*3 elements!'
        self.cmap = cmap

        self.logs_dir = os.path.join(flags.logs_dir, 'events')
        self.checkpoint_dir = os.path.join(flags.logs_dir, 'models')
        self.logs_image_train_dir = os.path.join(flags.logs_dir, 'images_train')
        self.logs_image_valid_dir = os.path.join(flags.logs_dir, 'images_valid')
        self.dir_check()

    def dir_check(self):
        print('checking directories.')
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.logs_image_train_dir):
            os.makedirs(self.logs_image_train_dir)
        if not os.path.exists(self.logs_image_valid_dir):
            os.makedirs(self.logs_image_valid_dir)

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

    def load_train_datum_batch(self, start, end, need_aug=False):
        batch_len = end - start
        train_paths_batch = self.train_paths[start:end]
        x_batch, y_batch = [], []
        for idx, train_path in enumerate(train_paths_batch):
            x = Image.open(train_path[0])
            y = Image.open(train_path[1])
            x = np.array(x.resize((self.image_height, self.image_width), resample=Image.BILINEAR))
            y = np.array(y.resize((self.image_height, self.image_width), resample=Image.NEAREST))

            if need_aug:
                x, y = self.data_augmentation(x=x, y=y)
            else:
                if idx >= batch_len // 2:
                    x = np.fliplr(x)
                    y = np.fliplr(y)
            if len(x.shape) < 3:
                x = np.dstack((x, x, x))

            x_batch.append(x)
            y_batch.append(y)

        return x_batch, y_batch

    def load_val_datum_batch(self, start, end):
        valid_paths_batch = self.val_paths[start:end]
        x_batch, y_batch = [], []
        for idx, valid_path in enumerate(valid_paths_batch):
            x = Image.open(valid_path[0])
            y = Image.open(valid_path[1])
            x = np.array(x.resize((self.image_height, self.image_width), resample=Image.BILINEAR))
            y = np.array(y.resize((self.image_height, self.image_width), resample=Image.NEAREST))
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
        padding = height - self.image_height
        if padding != 0:
            off_h = np.random.randint(0, padding)
            off_w = np.random.randint(0, padding)
        x = x[off_h:off_h+self.image_height, off_w:off_w+self.image_width, :]
        y = y[off_h:off_h+self.image_height, off_w:off_w+self.image_width]

        return x, y

    def data2record(self, name='coco_stuff2017_train.tfrecords', is_training=True, test_num=None):
        filename = os.path.join(self.TFRecord_dir, name)
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        if is_training:
            paths = self.train_paths
        else:
            paths = self.val_paths

        for idx, train_path in enumerate(paths):
            print('[{:d}/{:d}]'.format(idx, len(paths)))
            x = np.array(Image.open(train_path[0]))
            y = np.array(Image.open(train_path[1]))

            rows = x.shape[0]
            cols = x.shape[1]
            # Some images are gray-scale
            if len(x.shape) < 3:
                x = np.dstack((x, x, x))
            # Label [92, 183] -> [1, 92]
            # y[np.nonzero(y < 92)] = 183
            # y -= 91

            image_raw = x.tostring()
            label_raw = y.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'label_raw': _bytes_feature(label_raw),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())

            if test_num is not None and idx > test_num:
                break

        writer.close()

    def data2record_test(self, name='coco_stuff2017_test.tfrecords', is_dev=True, test_num=None):
        filename = os.path.join(self.TFRecord_dir, name)
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        if is_dev:
            coco_gt = COCO(self.annotations_test_dev_info)
        else:
            coco_gt = COCO(self.annotations_test_info)
        for key_idx, key in enumerate(coco_gt.imgs):
            print('{:d}/{:d}'.format(key_idx, len(coco_gt.imgs)))
            value = coco_gt.imgs[key]
            file_name = value['file_name']
            x = Image.open(os.path.join(self.images_test_dir, file_name))
            x = np.array(x)

            rows = x.shape[0]
            cols = x.shape[1]
            # Some images are gray-scale
            if len(x.shape) < 3:
                x = np.dstack((x, x, x))

            image_raw = x.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())

            if test_num is not None and key_idx > test_num:
                break

        writer.close()

    @staticmethod
    def preprocess_data(image, label):
        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.cast(image, tf.float32)
        image = image - 127.5
        # tf.image.per_image_standardization(image)

        label = tf.cast(label, tf.int32)
        return image, label

    @staticmethod
    def deprocess_data(image, label, pred_batches=None):
        image = image + 127.5
        label += 91
        label[label == 91] = 183
        if pred_batches is not None:
            pred_batches = np.argmax(pred_batches, axis=3) + 91
            pred_batches[pred_batches == 91] = 183
            return image, label, pred_batches
        else:
            return image, label

    def parse_record(self, record):
        features = tf.parse_single_example(
            record,
            # Defaults are not specified since both keys are required.
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label_raw': tf.FixedLenFeature([], tf.string)
            })

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, [height, width, 3])
        label = tf.decode_raw(features['label_raw'], tf.uint8)
        label = tf.reshape(label, [height, width, 1])

        # augmentation:
        # image = tf.cast(image, tf.float32)
        # label = tf.cast(label, tf.float32)
        # image = tf.image.random_brightness(image, max_delta=63)
        # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

        combined = tf.concat((image, label), axis=2)
        image_height = tf.maximum(height, self.image_height)
        image_width = tf.maximum(width, self.image_width)
        offset_height = tf.cast(tf.floor((image_height - height) / 2), tf.int32)
        offset_width = tf.cast(tf.floor((image_width - width) / 2), tf.int32)

        combined_pad = tf.image.pad_to_bounding_box(
            combined, offset_height, offset_width,
            image_height,
            image_width)
        combined_crop = tf.random_crop(value=combined_pad, size=(self.image_height, self.image_width, 4))
        combined_crop = tf.image.random_flip_left_right(combined_crop)
        # combined_crop = tf.image.crop_to_bounding_box(combined_pad, 0, 0, FLAGS.image_height, FLAGS.image_width)
        # combine = tf.image.resize_image_with_crop_or_pad(combine, FLAGS.image_height, FLAGS.image_width)
        # OPTIONAL: Could reshape into a 28x28 image and apply distortions
        # here.  Since we are not applying any distortions in this
        # example, and the next step expects the image to be flattened
        # into a vector, we don't bother.

        image = combined_crop[:, :, :3]
        label = combined_crop[:, :, -1]
        image, label = self.preprocess_data(image=image, label=label)

        return image, label

    def tfrecord_get_dataset(self, name, batch_size, shuffle_size=None):
        filename = os.path.join(self.TFRecord_dir, name)
        dataset = tf.contrib.data.TFRecordDataset(filename)
        dataset = dataset.map(self.parse_record)
        if shuffle_size is not None:
            dataset = dataset.shuffle(buffer_size=shuffle_size)
        dataset = dataset.batch(batch_size)
        return dataset

    def visualize_data(self, x_batches, y_batches, pred_batches, global_step, logs_dir):
        x_batches, y_batches, pred_batches, = self.deprocess_data(
            image=x_batches, label=y_batches, pred_batches=pred_batches)

        for batch_idx, x_batch in enumerate(x_batches):
            x_png = Image.fromarray(x_batch.astype(np.uint8)).convert('P')
            x_png.save('{}/{:d}_{:d}_0_rgb.png'.format(
                logs_dir, global_step, batch_idx), format='PNG')

        for batch_idx, y_batch in enumerate(y_batches):
            y_png = Image.fromarray(y_batch.astype(np.uint8)).convert('P')
            y_png.putpalette(list(self.cmap))
            y_png.save('{}/{:d}_{:d}_1_gt.png'.format(
                logs_dir, global_step, batch_idx), format='PNG')

        for batch_idx, pred_batch in enumerate(pred_batches):
            pred_png = Image.fromarray(pred_batch.astype(np.uint8)).convert('P')
            pred_png.putpalette(list(self.cmap))
            pred_png.save('{}/{:d}_{}_2_pred.png'.format(
                logs_dir, global_step, batch_idx), format='PNG')


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
            self.annotations_val_dir, self.images_val_dir)
            for annotations_val_path in self.annotations_val_paths]
        self.val_paths = list(zip(self.images_val_paths, self.annotations_val_paths))
        return self

    def load_train_paths(self):
        self.annotations_train_paths = sorted(glob(os.path.join(self.annotations_train_dir, "*.png")))
        self.images_train_paths = [annotations_train_path.replace(
            self.annotations_train_dir, self.images_train_dir)
            for annotations_train_path in self.annotations_train_paths]
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
