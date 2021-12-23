from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper
from math import *
from timeit import default_timer as timer

IMAGE_SIZE = [320, 320]
CUT_SIZE = (32, 32)
label2num_dic = {"0": 0,
                 "1": 1,
                 "2": 2,
                 "3": 3,
                 "4": 4,
                 "5": 5,
                 "6": 6,
                 "7": 7,
                 "8": 8,
                 "9": 9,
                 "A": 10,
                 "B": 11,
                 "C": 12,
                 "D": 13,
                 "E": 14,
                 "F": 15,
                 "G": 16,
                 "H": 17,
                 "J": 18,
                 "K": 19,
                 "P": 20,
                 "S": 21,
                 "R": 22,
                 "Q": 23}

num2label_dic = {"0": "0",
                 "1": "1",
                 "2": "2",
                 "3": "3",
                 "4": "4",
                 "5": "5",
                 "6": "6",
                 "7": "7",
                 "8": "8",
                 "9": "9",
                 "10": "A",
                 "11": "B",
                 "12": "C",
                 "13": "D",
                 "14": "E",
                 "15": "F",
                 "16": "G",
                 "17": "H",
                 "18": "J",
                 "19": "K",
                 "20": "P",
                 "21": "S",
                 "22": "R",
                 "23": "Q"}


class OcrInference(object):
    def __init__(self):

        db_model_path = '../pbMode/db_model.pb'
        dec_model_path = '../pbMode/dec_model.pb'

        self.sess = tf.Session()
        with gfile.FastGFile(db_model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        with gfile.FastGFile(dec_model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        self.sess.run(tf.global_variables_initializer())

        self.db_input_image = self.sess.graph.get_tensor_by_name('image_input:0')
        self.db_out = self.sess.graph.get_tensor_by_name('dbnet/proba3_sigmoid:0')
        self.dec_input_image = self.sess.graph.get_tensor_by_name('cut_image_input:0')
        self.decision_out = self.sess.graph.get_tensor_by_name('decision_out:0')
        self.decision_prob = self.sess.graph.get_tensor_by_name('decision_prob:0')

    def rectify_and_cut(self, image_path, max_boxes=2):
        try:
            image = cv2.imread(image_path)
        except:
            raise FileNotFoundError("{} file not found".format(image_path))

        start = timer()
        image = image.astype(np.float32)
        origin_image = image.copy()/255.0
        # scale_h = origin_image.shape[1]/IMAGE_SIZE[0]
        scale = origin_image.shape[1]/IMAGE_SIZE[1]

        image = self.resize_image(IMAGE_SIZE[0], image)
        image = np.array(image[np.newaxis, :, :, :])

        image = image / 255.0
        proba_map = self.sess.run(self.db_out, feed_dict={self.db_input_image: image})

        proba_map = np.squeeze(proba_map, 0)
        image = np.squeeze(image, 0)

        image *= 255.0
        bitmap = proba_map > 0.3
        boxes, scores, angles = self.polygons_from_bitmap(proba_map, bitmap, image.shape[0], image.shape[1], box_thresh=0.5)
        boxes = [z[0] for z in sorted(list(zip(boxes, scores)), key=lambda x:x[1], reverse=True)][:max_boxes]

        for i in range(len(boxes)):
            pt1, pt2, pt3, pt4 = boxes[i]
            height = origin_image.shape[0]  # 原始图像高度
            width = origin_image.shape[1]  # 原始图像宽度
            rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angles[i], 1)  # 计算旋转矩阵
            heightNew = int(width * fabs(sin(radians(angles[i]))) + height * fabs(cos(radians(angles[i]))))
            widthNew = int(height * fabs(sin(radians(angles[i]))) + width * fabs(cos(radians(angles[i]))))  #计算旋转后的图片尺寸

            rotateMat[0, 2] += (widthNew - width) / 2
            rotateMat[1, 2] += (heightNew - height) / 2
            imgRotation = cv2.warpAffine(origin_image, rotateMat, (widthNew, heightNew), borderValue=(0, 0, 0))  # 旋转图片
            # cv2.imshow('rotateImg2', imgRotation)
            # cv2.waitKey(0)

            [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]*scale], [pt1[1]*scale], [1]]))
            [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]*scale], [pt3[1]*scale], [1]]))
            [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]*scale], [pt2[1]*scale], [1]]))
            [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]*scale], [pt4[1]*scale], [1]]))   # 计算旋转后的矩形框坐标

            # 处理反转的情况
            if pt2[1] > pt4[1]:
                pt2[1], pt4[1] = pt4[1], pt2[1]
            if pt1[0] > pt3[0]:
                pt1[0], pt3[0] = pt3[0], pt1[0]

            imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
            if imgOut.shape[0] < imgOut.shape[1]:
                imgOut = cv2.rotate(imgOut, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # cv2.imshow("imgOut", imgOut)  # 裁减得到的旋转矩形框
            # cv2.waitKey(0)

            cv2.imwrite('../test_cut/{}-{}.jpg'.format(image_path.split('.')[0].split('/')[-1], str(i)), imgOut*255.0)
        end = timer()
        # print(end-start)

    def polygons_from_bitmap(self, pred, bitmap, dest_width, dest_height, max_candidates=100, box_thresh=0.7):
        pred = pred[..., 0]
        bitmap = bitmap[..., 0]
        height, width = bitmap.shape
        boxes = []
        scores = []
        angles = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:max_candidates]:
            epsilon = 0.01 * cv2.arcLength(contour, True)  # 轮廓计算周长(用于折线化)
            approx = cv2.approxPolyDP(contour, epsilon, True)  # 将连续曲线折线化
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))  # 计算contour所围区域在prob_map上的平局prob
            if box_thresh > score:
                continue
            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=2.0)  # 将contour进行一定的膨胀
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            rect, sside, angle = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < 5:  # 如果最小外接矩形的ymax<5则不要？
                continue

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(rect)  # box是以轮廓点集的格式呈现的(n,2)，点的个数n的个数不固定，2为x和y
            scores.append(score)
            angles.append(angle)
        return boxes, scores, angles

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)  # 找到contour的最小外接矩形（长, 宽, 旋转角度）
        angle = bounding_box[-1]
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])  # 生成外接矩形的四个顶点，并排序

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1].tolist(), points[index_2].tolist(),
               points[index_3].tolist(), points[index_4].tolist()]  # 转换为从左下点开始逆时针的四个顶点
        return box, min(bounding_box[1]), angle

    def box_score_fast(self, bitmap, _box):
        # 计算 box 包围的区域的平均 prob
        # return 之前只为生成一个与多边形box外接的矩形mask，mask矩阵中被box包含的区域为1，box之外的区域为0
        """ 如
            [[0 0 0  0 0 0]
             [0 0 0  0 0 0]
             [0 0 1  1 1 0]
             [0 1 1  1 0 0]
             [0 0 1  1 0 0]
             [0 0 0  0 0 0]]
        """
        # 最后再用此mask计算prob_map[ymin:ymax + 1, xmin:xmax + 1]的平均prob
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)

        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length if poly.area > 0 else 1
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def resize_image(self, size, image):
        h, w, c = image.shape
        scale_w = size / w
        scale_h = size / h
        scale = min(scale_w, scale_h)
        h = int(h * scale)
        w = int(w * scale)
        padimg = np.zeros((size, size, c), image.dtype)
        padimg[:h, :w] = cv2.resize(image, (w, h))

        return padimg


if __name__ == '__main__':
    label_dic = {}
    ocr = OcrInference()
    # image_root = 'E:/DATA/ocr_img/cut_image/'
    image_root = 'F:/NGST1/1/dn/'
    image_paths = [i[2] for i in os.walk(image_root)][0]
    start = timer()
    for path in image_paths:
        # ocr.cut_image(os.path.join(image_root, path), label_dic)
        ocr.rectify_and_cut(os.path.join(image_root, path))

    end = timer()
    print("total time for {} images, average {}s per image".format(len(image_paths), (end-start)/len(image_paths)))

