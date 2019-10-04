# encoding = utf-8
import numpy as np


class Config(object):

    # mean and std in RGB order.
    # Un-scaled version: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # PREPROC_PIXEL_MEAN = [123.675, 116.28, 103.53]
    # PREPROC_PIXEL_STD = [58.395, 57.12, 57.375]

    DATA_DIR = './data'

    ANCHOR_SCALES = [8, 16, 32]
    ANCHOR_RATIOS = [0.5, 1, 2]

    # Number of filters for the RPN layer, 512 for VGG16
    RPN_CHANNELS = 512


    """general configs"""

    IMAGE_SCALES = 600  # rescale后图像的短边
    IMAGE_MAX_SIZE = 1000  # rescale后的图像长边不超过1000像素

    # Pixel mean values (BGR order) as a (1, 1, 3) array
    # We use the same pixel mean for all networks even though it's not exactly what
    # they were trained with
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

    POOLING_SIZE = 7

    """train configs"""
    TRAIN_WEIGHT_DECAY = 0.0001

    BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
    BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

    # NMS
    TRAIN_RPN_POST_NMS_TOP_N = 2000
    TRAIN_RPN_NMS_THRESH = 0.7


    """test configs"""
    TEST_MODE = 'nms'
    TEST_BBOX_REG = True

    # NMS
    TEST_RPN_POST_NMS_TOP_N = 300  # nms后保留300个方框
    TEST_RPN_NMS_THRESH = 0.7  # nms IoU阈值为0.7


cfg = Config()



