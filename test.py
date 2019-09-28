# encoding = utf-8

from data_loader import get_voc_dataset
import cv2

if __name__ == '__main__':
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

