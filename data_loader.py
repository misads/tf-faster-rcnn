# encoding=utf-8

import tensorflow as tf
import numpy as np
from config import cfg


# 将图像减去均值，再除以标准差
def image_preprocess(image, bgr=True):
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
    pass
