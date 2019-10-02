# encoding = utf-8
class Config(object):

    # mean and std in RGB order.
    # Un-scaled version: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    PREPROC_PIXEL_MEAN = [123.675, 116.28, 103.53]
    PREPROC_PIXEL_STD = [58.395, 57.12, 57.375]

    DATA_DIR = './data'

    ANCHOR_SCALES = [8, 16, 32]
    ANCHOR_RATIOS = [0.5, 1, 2]

    # Number of filters for the RPN layer, 512 for VGG16
    RPN_CHANNELS = 512

    """train configs"""
    TRAIN_WEIGHT_DECAY = 0.0001


    """test configs"""
    TEST_MODE = 'nms'

cfg = Config()



