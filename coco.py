from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import scipy.misc
import dataset_parser
from PIL import Image
from pycocotools.coco import COCO

print('test')
# Define path
ann_file = './dataset/coco_stuff/annotations/image_info_test-dev2017.json'
test_dir = './dataset/coco_stuff/test2017'
# Initialize COCO ground truth API
coco_gt = COCO(ann_file)
for key_idx, key in enumerate(coco_gt.imgs):
    value = coco_gt.imgs[key]
    file_name = value['file_name']
    image = Image.open(os.path.join(test_dir, file_name))
    ##############################################################
    width, height = image.size
    pooling_size = 128
    width_new = ((width // pooling_size) + 1) * pooling_size if width % pooling_size != 0 else width
    height_new = ((height // pooling_size) + 1) * pooling_size if height % pooling_size != 0 else height

    new_im = Image.new("RGB", (width_new, height_new))
    box_left = np.floor((width_new - width) / 2).astype(np.int32)
    box_upper = np.floor((height_new - height) / 2).astype(np.int32)
    new_im.paste(image, (box_left, box_upper))
    image = new_im
    scipy.misc.imshow(image)

    pred_png = image.crop((box_left, box_upper, width, height))
    scipy.misc.imshow(pred_png)

    print('{:d}/{:d} height:{:d} width:{:d} height_new:{:d} width_new:{:d}'.format(
        key_idx, len(coco_gt.imgs), height, width, height_new, width_new))
    #image = image.resize((FLAGS.image_width, FLAGS.image_height), resample=Image.BILINEAR)
    ##############################################################
    if key_idx > 1:
        break
