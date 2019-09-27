# encoding = utf-8

import tensorflow as tf


class BaseModel(object):
    def __init__(self):
        pass

    def build_graph(self):
        self.image = tf.placeholder(tf.float32, shape=[1, None, None, 3])  # 一次只能输入一张3通道的图像
        self.im_info = tf.placeholder(tf.float32, shape=[3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])  # x,y,w,h,label




