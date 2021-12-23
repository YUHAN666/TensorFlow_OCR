from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper

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

    def cut_image(self, image_path, label_dict, max_boxes=2):
        try:
            image = cv2.imread(image_path)
        except:
            raise FileNotFoundError("{} file not found".format(image_path))
        image = image.astype(np.float32)
        origin_image = image.copy()
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
        boxes, scores = self.polygons_from_bitmap(proba_map, bitmap, image.shape[0], image.shape[1], box_thresh=0.5)
        boxes = [z[0] for z in sorted(list(zip(boxes, scores)), key=lambda x:x[1], reverse=True)][:max_boxes]
        img = image.copy()
        for box in boxes:
            cv2.drawContours(img, [np.array(box)], -1, (0, 255, 0), 2)
        label_str = ''
        coordinates = []
        for contour in boxes:
            contour = np.reshape(contour, (-1, 2))
            xmin = sorted(list(contour), key=lambda x: x[0])[0][0]
            xmax = sorted(list(contour), key=lambda x: x[0])[-1][0]
            ymin = sorted(list(contour), key=lambda x: x[1])[0][1]
            ymax = sorted(list(contour), key=lambda x: x[1])[-1][1]

            coordinates.append((xmin, xmax, ymin, ymax))
        coordinates = sorted(coordinates, key=lambda x: x[2])
        coordinates1 = sorted(coordinates[:5], key=lambda x: x[0])
        coordinates2 = sorted(coordinates[5:], key=lambda x: x[0])

        for xmin, xmax, ymin, ymax in coordinates1+coordinates2:

            cut = origin_image[int(ymin*scale):int(ymax*scale), int(xmin*scale):int(xmax*scale), :]

            cut_image = image[ymin:ymax, xmin:xmax, :]
            img = cut_image.copy()
            try:
                cut_image = cv2.resize(cut_image, CUT_SIZE)
            except:
                raise ValueError("The cut image is too small")


            cut_image = np.array(cut_image[np.newaxis, :, :])
            decision = self.sess.run(self.decision_out, feed_dict={self.dec_input_image: cut_image / 255.0})[0]
            label = num2label_dic[str(decision)]
            if label not in label_dic.keys():
                label_dic[label] = 1
            else:
                label_dic[label] += 1
            # cv2.putText(img, label, (xmin+5, ymin+20), 1, 1.0, (0, 0, 255))
            cv2.imwrite('../test_cut/{}-{}.jpg'.format(label, label_dic[label]), cut)

    def rectify_and_cut(self, image_path, max_boxes=2):
        try:
            image = cv2.imread(image_path)
        except:
            raise FileNotFoundError("{} file not found".format(image_path))
        image = image.astype(np.float32)
        origin_image = image.copy()
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
        boxes, scores = self.polygons_from_bitmap(proba_map, bitmap, image.shape[0], image.shape[1], box_thresh=0.5)
        boxes = [z[0] for z in sorted(list(zip(boxes, scores)), key=lambda x:x[1], reverse=True)][:max_boxes]
        img = image.copy()
        for box in boxes:
            cv2.drawContours(img, [np.array(box)], -1, (0, 255, 0), 2)
        label_str = ''
        coordinates = []
        for contour in boxes:
            contour = np.reshape(contour, (-1, 2))
            xmin = sorted(list(contour), key=lambda x: x[0])[0][0]
            xmax = sorted(list(contour), key=lambda x: x[0])[-1][0]
            ymin = sorted(list(contour), key=lambda x: x[1])[0][1]
            ymax = sorted(list(contour), key=lambda x: x[1])[-1][1]

            coordinates.append((xmin, xmax, ymin, ymax))
        coordinates = sorted(coordinates, key=lambda x: x[2])
        coordinates1 = sorted(coordinates[:5], key=lambda x: x[0])
        coordinates2 = sorted(coordinates[5:], key=lambda x: x[0])

        for xmin, xmax, ymin, ymax in coordinates1+coordinates2:

            cut = origin_image[int(ymin*scale):int(ymax*scale), int(xmin*scale):int(xmax*scale), :]

            cut_image = image[ymin:ymax, xmin:xmax, :]
            img = cut_image.copy()
            try:
                cut_image = cv2.resize(cut_image, CUT_SIZE)
            except:
                raise ValueError("The cut image is too small")


            cut_image = np.array(cut_image[np.newaxis, :, :])
            decision = self.sess.run(self.decision_out, feed_dict={self.dec_input_image: cut_image / 255.0})[0]
            label = num2label_dic[str(decision)]
            if label not in label_dic.keys():
                label_dic[label] = 1
            else:
                label_dic[label] += 1
            # cv2.putText(img, label, (xmin+5, ymin+20), 1, 1.0, (0, 0, 255))
            cv2.imwrite('../test_cut/{}-{}.jpg'.format(label, label_dic[label]), cut)

    def decision(self, image_path):

        try:
            image = cv2.imread(image_path)
        except:
            raise FileNotFoundError("{} file not found".format(image_path))
        image = image.astype(np.float32)
        image = self.resize_image(IMAGE_SIZE[0], image)
        image = np.array(image[np.newaxis, :, :, :])

        image = image / 255.0
        proba_map = self.sess.run(self.db_out, feed_dict={self.db_input_image: image})

        proba_map = np.squeeze(proba_map, 0)
        image = np.squeeze(image, 0)

        image *= 255.0
        bitmap = proba_map > 0.3
        boxes, scores = self.polygons_from_bitmap(proba_map, bitmap, image.shape[0], image.shape[1], box_thresh=0.5)
        label_str = ''
        coordinates = []
        for contour in boxes:
            contour = np.reshape(contour, (-1, 2))
            xmin = sorted(list(contour), key=lambda x: x[0])[0][0]
            xmax = sorted(list(contour), key=lambda x: x[0])[-1][0]
            ymin = sorted(list(contour), key=lambda x: x[1])[0][1]
            ymax = sorted(list(contour), key=lambda x: x[1])[-1][1]

            coordinates.append((xmin, xmax, ymin, ymax))
        coordinates = sorted(coordinates, key=lambda x: x[2])
        coordinates1 = sorted(coordinates[:5], key=lambda x: x[0])
        coordinates2 = sorted(coordinates[5:], key=lambda x: x[0])

        for xmin, xmax, ymin, ymax in coordinates1+coordinates2:

            cut_image = image[ymin:ymax, xmin:xmax, :]
            try:
                cut_image = cv2.resize(cut_image, CUT_SIZE)
            except:
                raise ValueError("The cut image is too small")

            cut_image = np.array(cut_image[np.newaxis, :, :, :])
            decision, prob = self.sess.run([self.decision_out, self.decision_prob], feed_dict={self.dec_input_image: cut_image / 255.0})
            if prob >= 0.5:
                label = num2label_dic[str(decision[0])]
                label_str += label
            else:
                print('{}:{}'.format(decision[0], prob[0]))
                # cv2.imshow("cut_image", np.squeeze(cut_image, axis=0)[:, :, 0]*255.0)
                # cv2.waitKey()
                cv2.imwrite('../wrong/{}-{}.jpg'.format(str(decision[0]), str(int(1000*prob[0]))), np.squeeze(cut_image, axis=0)[:, :, 0])
        return label_str

    def polygons_from_bitmap(self, pred, bitmap, dest_width, dest_height, max_candidates=100, box_thresh=0.7):
        pred = pred[..., 0]
        bitmap = bitmap[..., 0]
        height, width = bitmap.shape
        boxes = []
        scores = []

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
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < 5:  # 如果最小外接矩形的ymax<5则不要？
                continue

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())  # box是以轮廓点集的格式呈现的(n,2)，点的个数n的个数不固定，2为x和y
            scores.append(score)
        return boxes, scores

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)  # 找到contour的最小外接矩形（长, 宽, 旋转角度）
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

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]  # 转换为从左下点开始逆时针的四个顶点
        return box, min(bounding_box[1])

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
    for path in image_paths:
        # ocr.cut_image(os.path.join(image_root, path), label_dic)
        ocr.cut_image(os.path.join(image_root, path), label_dic)
