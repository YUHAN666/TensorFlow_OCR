import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


def polygons_from_bitmap(pred, bitmap, dest_width, dest_height, max_candidates=100, box_thresh=0.7):
    pred = pred[..., 0]
    bitmap = bitmap[..., 0]
    height, width = bitmap.shape
    boxes = []
    scores = []

    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours[:max_candidates]:
        epsilon = 0.01 * cv2.arcLength(contour, True)           # 轮廓计算周长(用于折线化)
        approx = cv2.approxPolyDP(contour, epsilon, True)       # 将连续曲线折线化

        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(pred, points.reshape(-1, 2))     # 计算contour所围区域在prob_map上的平局prob
        if box_thresh > score:
            continue
        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=2.0)              # 将contour进行一定的膨胀
            if len(box) > 1:
                continue
        else:
            continue
        box = box.reshape(-1, 2)
        _, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 5:       # 如果最小外接矩形的ymax<5则不要？
            continue

        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.tolist())          # box是以轮廓点集的格式呈现的(n,2)，点的个数n的个数不固定，2为x和y
        scores.append(score)
    return boxes, scores


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)                 # 找到contour的最小外接矩形（长, 宽, 旋转角度）
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])      # 生成外接矩形的四个顶点，并排序

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
           points[index_3], points[index_4]]            # 转换为从左下点开始逆时针的四个顶点
    return box, min(bounding_box[1])


def box_score_fast(bitmap, _box):
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


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length if poly.area > 0 else 1
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded
