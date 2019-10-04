# encoding = utf-8
from config import cfg
import numpy as np
import tensorflow as tf

from data_loader import get_voc_dataset
import cv2

from models.vgg16 import Vgg16
from data_loader import _get_blobs
from utils.bbox import _clip_boxes, bbox_transform_inv


def test_pic(sess, net, path, weights_filename, max_per_image=100, thresh=0.):
    num_classes = 21
    np.random.seed(cfg.RNG_SEED)
    """Test a Fast R-CNN network on an image database."""
    num_images = 1
    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    # output_dir = get_output_dir(imdb, weights_filename)
    # timers

    for i in range(num_images):
        im = cv2.imread(path)


        scores, boxes, rois, roi_scores, anchors = im_detect(sess, net, im)

        area = (rois[:,2] - rois[:,0]) / (rois[:,3] - rois[:,1])
        bbs = np.append(rois, 1-scores[:, 0:1], axis=1)
        bbs = bbs[np.argsort(bbs[:, 4])]
        bbs = bbs[::5, :]
        area = (anchors[:,2] - anchors[:,0]) * (anchors[:,3] - anchors[:,1])
        area.sort()
        print('anchors:', len(anchors))
        for a in area:
            print(a)
        #print('bbs', bbs)

        l = len(anchors) // 9
        anchors_s = anchors.copy()
        for a in range(9):
            anchors_s[a * l: a*l + l] = anchors[a::9]
        anchors_s = anchors_s[::10]
        #anchors = anchors[::9]
        for id in range(len(anchors_s)):
            r = anchors_s[id]
            img = im.copy()
            cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (255, 255, 255), 1)
            #cv2.putText(img, '%.3f' % (bbs[id, 4]), (r[0], int(r[1]) + 15), 0, 0.6, (0, 255, 0), 1)
            #cv2.imshow('image', img)
            cv2.imwrite('anchors/%05d.jpg' % id, img)
            # cv2.imwrite('result.jpg', im)
            #id = id + 1
            #cv2.waitKey(0)

        '''
        for id in range(len(bbs)):
            r = bbs[id]
            img = im.copy()
            cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 1)
            cv2.putText(img, '%.3f' % (bbs[id, 4]), (r[0], int(r[1]) + 15), 0, 0.6, (0, 255, 0), 1)
            cv2.imshow('image', img)
            cv2.imwrite('rois/%03d.jpg' % id, img)
            # cv2.imwrite('result.jpg', im)
            id = id + 1
            cv2.waitKey(0)
        '''
        print(roi_scores.shape)
        print('rois[0]', rois[1])
        print('roi_scores[0]', roi_scores[1])

        # skip j = 0, because it's the background class
        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            #keep = nms(cls_dets, cfg.TEST.NMS)
            #cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time))

        # pt1x, pt1y = 20, 20
        # pt2x, pt2y = 100, 100
        # cv2.rectangle(im, (pt1x, pt1y), (pt2x, pt2y), (255, 255, 255), 3)

        colors = [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 255, 255],
            [120, 0, 0],
            [0, 120, 0],
            [0, 0, 120],  # 10
            [120, 120, 0],
            [120, 0, 120],
            [36, 51, 138],
            [120, 120, 120],
            [120, 255, 0],
            [120, 0, 255],
            [0, 120, 255],
            [255, 120, 0],
            [255, 0, 120],
            [0, 255, 120],

        ]

        caption = [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
            '10',
            '11',
            'cow',
            'horse',
            '14',
            'person',
            '16',
            '17',
            '18',
            '19',
            '20',
            '21',

        ]
        '''
        for i in range(num_classes):
            for j in all_boxes[i]:
                for k in j:
                    if k[4] > 0.5:
                        cv2.rectangle(im, (k[0], k[1]), (k[2], k[3]), colors[i], 1)
                        cv2.putText(im, caption[i], (int(k[0] + 3), int(k[1] + 16)), 0, 0.8, colors[i], 1)
                        print(k)
        '''
        cv2.imshow('image', im)
        # cv2.imwrite('result.jpg', im)
        cv2.waitKey(0)

    # det_file = os.path.join(output_dir, 'detections.pkl')
    # with open(det_file, 'wb') as f:
    #    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)


def im_detect(sess, net, im):
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

    _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])

    boxes = rois[:, 1:5] / im_scales[0]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    if cfg.TEST_BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


if __name__ == '__main__':
    '''
    dataset = get_voc_dataset()
    for j in range(50):
        img = dataset.image_path_at(j)
        roi = dataset.roidb
        img = cv2.imread(img)
        boxes = roi[j]['boxes']
        classes = roi[j]['gt_classes']
        for i in range(len(boxes)):
            box = boxes[i]
            class_index = classes[i]
            label = dataset.classes[class_index]
            cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), (255, 255, 255), 1)
            cv2.putText(img, str(label), (box[0], box[1] + 16), 0, 0.6, (255, 255, 255), 1)
        cv2.imshow("result", img)
        #cv2.imwrite('pascol_voc.jpg', img)
        cv2.waitKey(0)
    '''

    dataset = get_voc_dataset()
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network

    net = Vgg16()

    # load model
    net.create_architecture(dataset.num_classes, training=False,
                            tag='default',
                            anchor_scales=cfg.ANCHOR_SCALES,
                            anchor_ratios=cfg.ANCHOR_RATIOS)

    # writer = tf.summary.FileWriter("logs/", sess.graph)


    saver = tf.train.Saver()
    model = 'output/vgg16_faster_rcnn_iter_70000.ckpt'
    saver.restore(sess, model)
    print('Checkpoint loaded.')

    #sess.run(tf.global_variables_initializer())

    im = cv2.imread('a.jpg')
    '''
    blobs, im_scales = _get_blobs(im)
    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    '''
    #net.debug(sess, blobs['data'], blobs['im_info'])

    scores, boxes = im_detect(sess, net, im)

    print(scores.shape)
    print(boxes.shape)

    num_classes = dataset.num_classes

    all_boxes = [[] for _ in range(num_classes)]

    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > 0.)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        #keep = nms(cls_dets, cfg.TEST.NMS)
        #cls_dets = cls_dets[keep, :]
        all_boxes[j] = cls_dets



    image_scores = np.hstack([all_boxes[j][:, -1]
                              for j in range(1, num_classes)])
    #if len(image_scores) > max_per_image:
    #image_thresh = np.sort(image_scores)[-20]
    image_thresh = 0.965
    for j in range(1, num_classes):
        keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
        all_boxes[j] = all_boxes[j][keep, :]
        print(all_boxes[j].shape)

    """
    tf.image.nms????????????????
    """

    colors = [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255],
        [120, 0, 0],
        [0, 120, 0],
        [0, 0, 120],  # 10
        [120, 120, 0],
        [120, 0, 120],
        [36, 51, 138],
        [120, 120, 120],
        [120, 255, 0],
        [120, 0, 255],
        [0, 120, 255],
        [255, 120, 0],
        [255, 0, 120],
        [0, 255, 120],
    ]

    for i in range(1, num_classes):
        for j in all_boxes[i]:
            if j[4] > 0.5:
                cv2.rectangle(im, (j[0], j[1]), (j[2], j[3]), colors[i], 1)
                cv2.putText(im, dataset.classes[i] + '=%.3f' % j[4], (int(j[0] + 3), int(j[1] + 16)), 0, 0.6, colors[i], 1)
    cv2.imshow('image', im)
    # cv2.imwrite('result.jpg', im)
    cv2.waitKey(0)

    #test_net(sess, net, imdb, filename, max_per_image=args.max_per_image)
    #test_pic(sess, net, 'a.jpg', None, max_per_image=20)
    sess.close()
