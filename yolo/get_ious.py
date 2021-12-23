import math
import tensorflow as tf


def get_iou(pred_box, true_box):

	pred_box = tf.expand_dims(pred_box, axis=-2)
	pred_wh = pred_box[...,2:4]
	pred_xy = pred_box[...,:2]
	pred_min = pred_xy - pred_wh/2.
	pred_max = pred_xy + pred_wh/2.

	true_box = tf.expand_dims(true_box, axis=0)
	true_wh = true_box[...,2:4]
	true_xy = true_box[...,:2]
	true_min = true_xy - true_wh/2.
	true_max = true_xy + true_wh/2.

	iou_min = tf.maximum(pred_min, true_min)
	iou_max = tf.minimum(pred_max, true_max)

	iou_wh = tf.maximum(iou_max-iou_min, 0)
	iou_area = iou_wh[...,-1] * iou_wh[...,-2]
	pred_area = pred_wh[...,-1] * pred_wh[...,-2]
	true_area = true_wh[...,-1] * true_wh[...,-2]

	iou = iou_area / (pred_area + true_area - iou_area + 1e-3)

	return iou


def get_giou(pred_box, true_box):

	pred_box = tf.expand_dims(pred_box, axis=-2)
	pred_wh = pred_box[...,2:4]
	pred_xy = pred_box[...,:2]
	pred_min = pred_xy - pred_wh/2.
	pred_max = pred_xy + pred_wh/2.

	true_box = tf.expand_dims(true_box, axis=0)
	true_wh = true_box[...,2:4]
	true_xy = true_box[...,:2]
	true_min = true_xy - true_wh/2.
	true_max = true_xy + true_wh/2.

	iou_min = tf.maximum(pred_min, true_min)
	iou_max = tf.minimum(pred_max, true_max)

	iou_wh = tf.maximum(iou_max-iou_min, 0)
	iou_area = iou_wh[...,-1] * iou_wh[...,-2]
	pred_area = pred_wh[...,-1] * pred_wh[...,-2]
	true_area = true_wh[...,-1] * true_wh[...,-2]

	iou = iou_area / (pred_area + true_area - iou_area + 1e-3)

	giou_max = tf.maximum(pred_max, true_max)
	giou_min = tf.minimum(pred_min, true_min)
	giou_wh = tf.maximum(giou_max - giou_min, 0)
	giou_area = giou_wh[..., 0] * giou_wh[..., 1]

	giou = iou - (giou_area-iou_area)/giou_area

	return giou


def get_diou(pred_box, true_box):

	pred_box = tf.expand_dims(pred_box, axis=-2)
	pred_wh = pred_box[...,2:4]
	pred_xy = pred_box[...,:2]
	pred_min = pred_xy - pred_wh/2.
	pred_max = pred_xy + pred_wh/2.

	true_box = tf.expand_dims(true_box, axis=0)
	true_wh = true_box[...,2:4]
	true_xy = true_box[...,:2]
	true_min = true_xy - true_wh/2.
	true_max = true_xy + true_wh/2.

	iou_min = tf.maximum(pred_min, true_min)
	iou_max = tf.minimum(pred_max, true_max)

	iou_wh = tf.maximum(iou_max-iou_min, 0)
	iou_area = iou_wh[...,-1] * iou_wh[...,-2]
	pred_area = pred_wh[...,-1] * pred_wh[...,-2]
	true_area = true_wh[...,-1] * true_wh[...,-2]
	iou = iou_area / (pred_area + true_area - iou_area + 1e-3)

	diou_max = tf.maximum(pred_max, true_max)
	diou_min = tf.minimum(pred_min, true_min)
	diou = iou - tf.reduce_sum(tf.square(pred_xy-true_xy), axis=-1)\
	       /tf.reduce_sum(tf.square(diou_max-diou_min), axis=-1)

	return diou


def get_ciou(pred_box, true_box):

	pred_box = tf.expand_dims(pred_box, axis=-2)
	pred_wh = pred_box[...,2:4]
	pred_xy = pred_box[...,:2]
	pred_min = pred_xy - pred_wh/2.
	pred_max = pred_xy + pred_wh/2.

	true_box = tf.expand_dims(true_box, axis=0)
	true_wh = true_box[...,2:4]
	true_xy = true_box[...,:2]
	true_min = true_xy - true_wh/2.
	true_max = true_xy + true_wh/2.

	iou_min = tf.maximum(pred_min, true_min)
	iou_max = tf.minimum(pred_max, true_max)

	iou_wh = tf.maximum(iou_max-iou_min, 0)
	iou_area = iou_wh[...,-1] * iou_wh[...,-2]
	pred_area = pred_wh[...,-1] * pred_wh[...,-2]
	true_area = true_wh[...,-1] * true_wh[...,-2]
	iou = iou_area / (pred_area + true_area - iou_area + 1e-3)

	diou_max = tf.maximum(pred_max, true_max)
	diou_min = tf.minimum(pred_min, true_min)
	diou = iou - tf.reduce_sum(tf.square(pred_xy-true_xy), axis=-1)\
	       /tf.reduce_sum(tf.square(diou_max-diou_min), axis=-1)
	v = tf.constant(4.0/math.pi, dtype=true_wh[...,0].dtype)* tf.square(tf.atan(true_wh[...,0]/true_wh[...,1]) - tf.atan(pred_wh[...,0]/pred_wh[..., 1]))
	alpha = v/(1-iou+v)

	ciou = diou - alpha * v

	return ciou


def nms_loop_cond(pred_boxes, pred_scores, nms_boxes, nms_scores, iou_threshold, i):

	return i < nms_boxes.shape[0]

def nms_loop_body(pred_boxes, pred_scores, nms_boxes, nms_scores, iou_threshold, i):

	max_idx = tf.arg_max(pred_scores,dimension=-2)

	max_box = pred_boxes[max_idx:max_idx+1, ...]
	max_score = pred_scores[max_idx:max_idx+1, ...]

	nms_boxes[i, ...] = pred_boxes[max_idx:max_idx+1, ...]
	nms_scores[i, ...] = pred_scores[max_idx:max_idx+1, ...]
	i += 1

	pred_boxes = tf.concat([pred_boxes[:max_idx, ...], pred_boxes[max_idx+1:, ...]], axis=0)

	ious = get_iou(pred_boxes, max_box)

	iou_mask = ious < iou_threshold

	pred_boxes = tf.boolean_mask(pred_boxes, iou_mask)
	pred_scores = tf.boolean_mask(pred_scores, iou_mask)

	return pred_boxes, pred_scores, nms_boxes, nms_scores, iou_threshold, i


def get_iou_with_coordinates(pred_box, true_box):
	pred_box = tf.expand_dims(pred_box, axis=-2)
	pred_min = pred_box[..., :2]
	pred_max = pred_box[..., 2:4]
	pred_wh = tf.maximum(pred_max - pred_min, 0)

	true_box = tf.expand_dims(true_box, axis=0)
	true_min = true_box[..., :2]
	true_max = true_box[..., 2:4]
	true_wh = tf.maximum(true_max - true_min, 0)

	iou_min = tf.maximum(pred_min, true_min)
	iou_max = tf.minimum(pred_max, true_max)

	iou_wh = tf.maximum(iou_max - iou_min, 0)
	iou_area = iou_wh[..., -1] * iou_wh[..., -2]
	pred_area = pred_wh[..., -1] * pred_wh[..., -2]
	true_area = true_wh[..., -1] * true_wh[..., -2]

	iou = iou_area / (pred_area + true_area - iou_area + 1e-3)

	return iou
