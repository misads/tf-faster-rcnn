# encoding = utf-8

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

from config import cfg
from utils.anchors import generate_anchors_pre
from utils.layer_utils import proposal_layer_tf


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
        self.tag = tag
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios


        self.num_scales = len(anchor_scales)
        self.num_ratios = len(anchor_ratios)

        self.num_anchors = self.num_scales * self.num_ratios  # 9

        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN_WEIGHT_DECAY)
        # 使用arg_scope减少代码重复
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=None,
                       biases_initializer=tf.constant_initializer(0.0)):

            rois, cls_prob, bbox_pred = self._build_graph(training)

        layers_to_output = {'rois': rois}

        if training:
            pass
        else:
            # if test
            stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), self.num_classes)
            means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), self.num_classes)
            self.predictions["bbox_pred"] *= stds
            self.predictions["bbox_pred"] += means

        return layers_to_output

    def _build_graph(self, training=True):
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
            if cfg.TEST_MODE == 'nms':
                rois, roi_scores = self.proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.TEST_MODE == 'top':
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

    def reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe,
                                  tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores = proposal_top_layer_tf(
                    rpn_cls_prob,
                    rpn_bbox_pred,
                    self._im_info,
                    self._feat_stride,
                    self._anchors,
                    self._num_anchors
                )
            else:
                rois, rpn_scores = tf.py_func(proposal_top_layer,
                                              [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                               self._feat_stride, self._anchors, self._num_anchors],
                                              [tf.float32, tf.float32], name="proposal_top")

            rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
            rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

        return rois, rpn_scores

    def proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            rois, rpn_scores = proposal_layer_tf(
                rpn_cls_prob,
                rpn_bbox_pred,
                self._im_info,
                self._mode,
                self._feat_stride,
                self._anchors,
                self._num_anchors
            )


            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

        # Only use it if you have roi_pooling op written in tf.image

    def roi_pool_layer(self, bootom, rois, name):
        with tf.variable_scope(name) as scope:
            return tf.image.roi_pooling(bootom, rois,
                                        pooled_height=cfg.POOLING_SIZE,
                                        pooled_width=cfg.POOLING_SIZE,
                                        spatial_scale=1. / 16.)[0]

    def crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bounding boxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.POOLING_SIZE * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                             name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    def dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)

    def anchor_target_layer(self, rpn_cls_score, name):
        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32],
                name="anchor_target")

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                name="proposal_target")

            rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
            roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
            labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
            bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores
