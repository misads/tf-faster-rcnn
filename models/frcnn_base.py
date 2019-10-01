# encoding = utf-8

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

from config import cfg
from utils.anchors import generate_anchors_pre


class BaseModel(object):
    def __init__(self):
        self.image = None
        self.im_info = None
        self.gt_boxes = None
        self.num_classes = 0
        self.anchors = None
        self.anchor_length = 0

        self.predictions = {}
        self.losses = {}
        self.anchor_targets = {}
        self.proposal_targets = {}
        self.layers = {}
        self.gt_image = None
        self.act_summaries = []
        self.score_summaries = {}
        self.train_summaries = []
        self.event_summaries = {}
        self.variables_to_fix = {}

    def create_architecture(self, num_classes, training=True, tag='default',
                            anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):

        assert num_classes != 0, '[error] num_classes = 0'

        # placeholders
        self.image = tf.placeholder(tf.float32, shape=[1, None, None, 3])  # 一次只能输入一张3通道的图像
        self.im_info = tf.placeholder(tf.float32, shape=[3])  # height, width, scale
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])  # x,y,w,h,label

        self.num_classes = num_classes

        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN_WEIGHT_DECAY)
        # 使用arg_scope减少代码重复
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=None,
                       biases_initializer=tf.constant_initializer(0.0)):

            rois, cls_prob, bbox_pred = self.build_graph(training)

        if training:
            pass
        else:
            # if test
            stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), self.num_classes)
            means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), self.num_classes)
            self.predictions["bbox_pred"] *= stds
            self.predictions["bbox_pred"] += means

    def build_graph(self, training=True):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        net_conv = self.image_to_head(training)
        with tf.variable_scope(self.scope, self.scope):
            # build the anchors for the image
            self.anchor_component()
            # region proposal network
            rois, roi_scores = self.region_proposal(net_conv, training, initializer)
            # region of interest pooling
            pool5 = self.crop_pool_layer(net_conv, rois, "pool5")

        fc7 = self.head_to_tail(pool5, training)
        with tf.variable_scope(self.scope, self.scope):
            # region classification
            cls_prob, bbox_pred = self.region_classification(fc7, training,
                                                             initializer, initializer_bbox)

        self.score_summaries.update(self.predictions)

        return rois, cls_prob, bbox_pred

    def anchor_component(self):
        with tf.variable_scope('ANCHOR_' + self.tag) as scope:
            # just to get the shape right
            height = tf.to_int32(tf.ceil(self.im_info[0] / np.float32(self.feat_stride[0])))
            width = tf.to_int32(tf.ceil(self.im_info[1] / np.float32(self.feat_stride[0])))

            anchors, anchor_length = generate_anchors_pre(
                height,
                width,
                self.feat_stride,
                self.anchor_scales,
                self.anchor_ratios
            )

            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self.anchors = anchors
            self.anchor_length = anchor_length

    def region_proposal(self, net_conv, is_training, initializer):
        rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                          scope="rpn_conv/3x3")
        self.act_summaries.append(rpn)
        rpn_cls_score = slim.conv2d(rpn, self.num_anchors * 2, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_cls_score')
        # change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self.softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
        rpn_cls_prob = self.reshape_layer(rpn_cls_prob_reshape, self.num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = slim.conv2d(rpn, self.num_anchors * 4, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        if is_training:
            rois, roi_scores = self.proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            rpn_labels = self.anchor_target_layer(rpn_cls_score, "anchor")
            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_labels]):
                rois, roi_scores = self.proposal_target_layer(rois, roi_scores, "rpn_rois")
        else:
            if cfg.TEST.MODE == 'nms':
                rois, roi_scores = self.proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.TEST.MODE == 'top':
                rois, roi_scores = self.proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError

        self.predictions["rpn_cls_score"] = rpn_cls_score
        self.predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self.predictions["rpn_cls_prob"] = rpn_cls_prob
        self.predictions["rpn_cls_pred"] = rpn_cls_pred
        self.predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self.predictions["rois"] = rois
        self.predictions["roi_scores"] = rois

        return rois, roi_scores

    def region_classification(self, fc7, is_training, initializer, initializer_bbox):
        cls_score = slim.fully_connected(fc7, self.num_classes,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         activation_fn=None, scope='cls_score')
        cls_prob = self.softmax_layer(cls_score, "cls_prob")
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        bbox_pred = slim.fully_connected(fc7, self.num_classes * 4,
                                         weights_initializer=initializer_bbox,
                                         trainable=is_training,
                                         activation_fn=None, scope='bbox_pred')

        self.predictions["cls_score"] = cls_score
        self.predictions["cls_pred"] = cls_pred
        self.predictions["cls_prob"] = cls_prob
        self.predictions["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred

    def _image_to_head(self, is_training, reuse=None):
        raise NotImplementedError

    def _head_to_tail(self, pool5, is_training, reuse=None):
        raise NotImplementedError
