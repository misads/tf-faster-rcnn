# encoding=utf-8

import tensorflow as tf
import numpy as np
from config import cfg
from datasets.pascal_voc import pascal_voc

__sets = {}

for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        # 不使用标记为difficulty的图像
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, use_diff=False))


def image_preprocess(image, bgr=True):
    # 将图像减去均值，再除以标准差
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        mean = cfg.PREPROC_PIXEL_MEAN
        std = np.asarray(cfg.PREPROC_PIXEL_STD)
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        image = (image - image_mean) * image_invstd
        return image


def get_voc_dataset(dataset_name='voc_2007_trainval'):
    return __sets[dataset_name]()


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob
