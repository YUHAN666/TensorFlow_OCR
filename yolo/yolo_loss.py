import math
import numpy as np
import tensorflow as tf

from yolo.get_ious import get_iou, get_ciou


def yolo_loss(gt_box, pred_box, anchors, ignore_thresh=0.7, iou_loss=True):

	num_layers = len(anchors) // 3
	anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]

	input_shape = tf.cast(tf.shape(pred_box[0])[1:3] * 32, pred_box[0].dtype)
	num_class = tf.shape(gt_box[0])[-1] - 5

	loss = 0
	for l in range(num_layers):

		feature_shape = tf.cast(tf.shape(pred_box[l])[1:3], pred_box[l].dtype)
		grid_shape = tf.shape(pred_box[l])[1:3]
		grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
		grid_y = tf.tile(tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
		grid = tf.cast(tf.concat([grid_x, grid_y], axis=-1), pred_box[0].dtype)

		gt_xy = gt_box[l][..., :2] * feature_shape[::-1] - grid
		# gt_xy = tf.Print(gt_xy, [tf.reduce_max(gt_xy)], message='Debug gt_xy:')
		gt_wh = gt_box[l][..., 2:4] * input_shape[::-1] / anchors[anchor_mask[l]]
		# gt_wh = tf.Print(gt_wh, [tf.reduce_max(gt_wh)], message='Debug gt_wh:')
		gt_wh = tf.where(gt_wh > 0, tf.log(gt_wh), tf.zeros_like(gt_wh))

		gt_class_prob = gt_box[l][...,5:]
		object_mask = gt_box[l][..., 4:5]

		box_loss_scale = 2 - gt_box[l][..., 2:3] * gt_box[l][..., 3:4]

		reshaped_pred = tf.reshape(pred_box[l], (-1, grid_shape[0], grid_shape[1], 3, 5+num_class))
		# rect_pred_xy = (tf.sigmoid(reshaped_pred[...,:2])+grid)/tf.cast(grid_shape[::-1], pred_box[l].dtype)
		# rect_pred_wh = tf.exp(reshaped_pred[...,2:4]*anchors[anchor_mask[l]]/input_shape[::-1])
		rect_pred_xy, rect_pred_wh, _, _ = feat2gt(pred_box[l], anchors[anchor_mask[l]], input_shape, num_class)
		rect_pred = tf.concat([rect_pred_xy, rect_pred_wh], axis=-1)

		ignore_mask = tf.TensorArray(gt_box[0].dtype, size=1, dynamic_size=True)
		object_mask_bool = tf.cast(object_mask, 'bool')
		batch_size_int = tf.shape(pred_box[0])[0]
		batch_size_float = tf.cast(batch_size_int, pred_box[l].dtype)
		def loop_body(b, ignore_mask):
			true_box = tf.boolean_mask(gt_box[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
			# 计算batch中第b个图片的第l个feature map的所有gt_box和pred_box的iou（有feature_map.shape[0]*feature_map.shape[1]*3个pred_box）,
			# 所以共有feature_map.shape[0]*feature_map.shape[1]*3*gt_box.number的输出
			iou = get_ciou(rect_pred[b], true_box)  # yolov4使用ciou
			best_iou = tf.reduce_max(iou, axis=-1)
			# 如果第b个图片的第l个feature map的全部anchor与gt_box的iou均小于ignore_thresh，则在计算confidence_loss忽略该层
			ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_thresh, gt_box[l].dtype))
			return b + 1, ignore_mask

		# _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b<m, loop_body, [0, ignore_mask])
		_, ignore_mask = tf.while_loop(lambda b, *args: b < batch_size_int, loop_body, [0, ignore_mask])
		ignore_mask = ignore_mask.stack()
		ignore_mask = tf.expand_dims(ignore_mask, -1)


		# confidence_loss分为正loss和负loss
		# 正loss为计算gt_box位置的loss
		# 负loss为计算没有gt_box位置的loss位置的loss，如果某层所有anchor与gt_box的iou均小于阈值，则不计算该层的负loss
		confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=reshaped_pred[..., 4:5]) + \
		                  (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=reshaped_pred[..., 4:5]) * ignore_mask
		class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_class_prob, logits=reshaped_pred[..., 5:])
		confidence_loss = tf.reduce_sum(confidence_loss) / batch_size_float
		class_loss = tf.reduce_sum(class_loss) / batch_size_float

		if iou_loss:
			ciou_loss = get_ciou_loss(rect_pred, gt_box[l], box_loss_scale)
			ciou_loss = tf.reduce_sum(ciou_loss) / batch_size_float

			# ciou_loss = tf.Print(ciou_loss, [ciou_loss], message='Debug ciou loss:')
			# confidence_loss = tf.Print(confidence_loss, [confidence_loss], message='Debug confidence loss:')
			# class_loss = tf.Print(class_loss, [class_loss], message='Debug class_loss loss:')

			loss += ciou_loss + confidence_loss + class_loss
		else:
			xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_xy, logits=reshaped_pred[..., 0:2])
			wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(gt_wh - reshaped_pred[..., 2:4])
			xy_loss = tf.reduce_sum(xy_loss) / batch_size_float
			wh_loss = tf.reduce_sum(wh_loss) / batch_size_float

			# xy_loss = tf.Print(xy_loss, [xy_loss], message='Debug xy_loss loss:')
			# wh_loss = tf.Print(wh_loss, [wh_loss], message='Debug wh_loss loss:')
			# confidence_loss = tf.Print(confidence_loss, [confidence_loss], message='Debug confidence loss:')
			# class_loss = tf.Print(class_loss, [class_loss], message='Debug class_loss loss:')
			loss += xy_loss + wh_loss + confidence_loss + class_loss

	# xy_loss = tf.Print(xy_loss, [xy_loss], message='Debug xy_loss loss:')
	# wh_loss = tf.Print(wh_loss, [wh_loss], message='Debug wh_loss loss:')

	# loss = tf.Print(loss, [loss], message='Debug loss:')
	# loss = tf.Print(loss, [loss], message='Debug loss:')

	return loss


def yolo_loss_customized(gt_box, pred_box, anchors, ignore_thresh=0.7, iou_loss=True, down_scale=32):
	num_layers = 2
	anchor_mask = [[2, 3], [0, 1]]
	num_anchors = len(anchors)

	input_shape = tf.cast(tf.shape(pred_box[0])[1:3] * down_scale, pred_box[0].dtype)
	num_class = tf.shape(gt_box[0])[-1] - 5

	loss = 0
	for l in range(num_layers):

		feature_shape = tf.cast(tf.shape(pred_box[l])[1:3], pred_box[l].dtype)
		grid_shape = tf.shape(pred_box[l])[1:3]
		grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
		grid_y = tf.tile(tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
		grid = tf.cast(tf.concat([grid_x, grid_y], axis=-1), pred_box[0].dtype)

		gt_xy = gt_box[l][..., :2] * feature_shape[::-1] - grid
		# gt_xy = tf.Print(gt_xy, [tf.reduce_max(gt_xy)], message='Debug gt_xy:')
		gt_wh = gt_box[l][..., 2:4] * input_shape[::-1] / anchors[anchor_mask[l]]
		# gt_wh = tf.Print(gt_wh, [tf.reduce_max(gt_wh)], message='Debug gt_wh:')
		gt_wh = tf.where(gt_wh > 0, tf.log(gt_wh), tf.zeros_like(gt_wh))

		gt_class_prob = gt_box[l][..., 5:]
		object_mask = gt_box[l][..., 4:5]

		box_loss_scale = 2 - gt_box[l][..., 2:3] * gt_box[l][..., 3:4]

		reshaped_pred = tf.reshape(pred_box[l], (-1, grid_shape[0], grid_shape[1], num_anchors // 2, 5 + num_class))
		# rect_pred_xy = (tf.sigmoid(reshaped_pred[...,:2])+grid)/tf.cast(grid_shape[::-1], pred_box[l].dtype)
		# rect_pred_wh = tf.exp(reshaped_pred[...,2:4]*anchors[anchor_mask[l]]/input_shape[::-1])
		rect_pred_xy, rect_pred_wh, _, _ = feat2gt(pred_box[l], anchors[anchor_mask[l]], input_shape, num_class)
		rect_pred = tf.concat([rect_pred_xy, rect_pred_wh], axis=-1)

		ignore_mask = tf.TensorArray(gt_box[0].dtype, size=1, dynamic_size=True)
		object_mask_bool = tf.cast(object_mask, 'bool')
		batch_size_int = tf.shape(pred_box[0])[0]
		batch_size_float = tf.cast(batch_size_int, pred_box[l].dtype)

		def loop_body(b, ignore_mask):
			true_box = tf.boolean_mask(gt_box[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
			# 计算batch中第b个图片的第l个feature map的所有gt_box和pred_box的iou（有feature_map.shape[0]*feature_map.shape[1]*3个pred_box）,
			# 所以共有feature_map.shape[0]*feature_map.shape[1]*3*gt_box.number的输出
			iou = get_iou(rect_pred[b], true_box)  # yolov4使用ciou
			best_iou = tf.reduce_max(iou, axis=-1)
			# 如果第b个图片的第l个feature map的全部anchor与gt_box的iou均小于ignore_thresh，则在计算confidence_loss忽略该层
			ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_thresh, gt_box[l].dtype))
			return b + 1, ignore_mask

		# _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b<m, loop_body, [0, ignore_mask])
		_, ignore_mask = tf.while_loop(lambda b, *args: b < batch_size_int, loop_body, [0, ignore_mask])
		ignore_mask = ignore_mask.stack()
		ignore_mask = tf.expand_dims(ignore_mask, -1)

		# confidence_loss分为正loss和负loss
		# 正loss为计算gt_box位置的loss
		# 负loss为计算没有gt_box位置的loss位置的loss，如果某层所有anchor与gt_box的iou均小于阈值，则不计算该层的负loss
		confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
		                                                                        logits=reshaped_pred[..., 4:5]) + \
		                  (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
		                                                                              logits=reshaped_pred[...,
		                                                                                     4:5]) * ignore_mask
		class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_class_prob,
		                                                                   logits=reshaped_pred[..., 5:])
		confidence_loss = tf.reduce_sum(confidence_loss) / batch_size_float
		class_loss = tf.reduce_sum(class_loss) / batch_size_float

		if iou_loss:
			ciou_loss = get_ciou_loss(rect_pred, gt_box[l], box_loss_scale)
			ciou_loss = tf.reduce_sum(ciou_loss) / batch_size_float

			# ciou_loss = tf.Print(ciou_loss, [ciou_loss], message='Debug ciou loss:')
			# confidence_loss = tf.Print(confidence_loss, [confidence_loss], message='Debug confidence loss:')
			# class_loss = tf.Print(class_loss, [class_loss], message='Debug class_loss loss:')

			loss += ciou_loss + confidence_loss + class_loss
		else:
			xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_xy,
			                                                                                 logits=reshaped_pred[...,
			                                                                                        0:2])
			wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(gt_wh - reshaped_pred[..., 2:4])
			xy_loss = tf.reduce_sum(xy_loss) / batch_size_float
			wh_loss = tf.reduce_sum(wh_loss) / batch_size_float

			# xy_loss = tf.Print(xy_loss, [xy_loss], message='Debug xy_loss loss:')
			# wh_loss = tf.Print(wh_loss, [wh_loss], message='Debug wh_loss loss:')
			# confidence_loss = tf.Print(confidence_loss, [confidence_loss], message='Debug confidence loss:')
			# class_loss = tf.Print(class_loss, [class_loss], message='Debug class_loss loss:')
			loss += xy_loss + wh_loss + confidence_loss + class_loss



		# xy_loss = tf.Print(xy_loss, [xy_loss], message='Debug xy_loss loss:')
		# wh_loss = tf.Print(wh_loss, [wh_loss], message='Debug wh_loss loss:')


		# loss = tf.Print(loss, [loss], message='Debug loss:')
		# loss = tf.Print(loss, [loss], message='Debug loss:')

	return loss

def feat2gt(feats, anchors, input_shape, num_classes):
	num_anchors = anchors.shape[0]
	anchor_tensor = tf.reshape(tf.constant(anchors, dtype=feats.dtype), [1, 1, 1, num_anchors, 2])
	feats_w = tf.shape(feats)[2]
	feats_h = tf.shape(feats)[1]
	grid_x = tf.tile(tf.reshape(tf.range(0, feats_w), (1, -1, 1, 1)), [feats_h, 1, 1, 1])
	grid_y = tf.tile(tf.reshape(tf.range(0, feats_h), (-1, 1, 1, 1)), [1, feats_w, 1, 1])
	grid = tf.cast(tf.concat([grid_x, grid_y], axis=-1), dtype=feats.dtype)

	feats = tf.reshape(feats, (-1, feats_h, feats_w, num_anchors, 5 + num_classes))
	feats_xy = tf.sigmoid(feats[..., 0:2])
	feats_wh = feats[..., 2:4]
	gt_xy = (feats_xy + grid) / tf.cast(tf.shape(feats)[1:3][::-1], feats.dtype)
	gt_wh = tf.exp(feats_wh) * anchor_tensor / input_shape[::-1]

	gt_confidence = tf.sigmoid(feats[..., 4:5])
	gt_classes = tf.sigmoid(feats[..., 5:])

	return gt_xy, gt_wh, gt_confidence, gt_classes

def get_ciou_loss(pred_box, true_box, box_loss_scale):

	pred_xy = pred_box[...,:2]
	pred_wh = pred_box[...,2:4]
	pred_xmin_ymin = pred_xy - pred_wh/2
	pred_xmax_ymax = pred_xy + pred_wh/2

	true_xy = true_box[...,:2]
	true_wh = true_box[...,2:4]
	true_xmin_ymin = true_xy - true_wh/2
	true_xmax_ymax = true_xy + true_wh/2

	intersect_xmin_ymin = tf.maximum(pred_xmin_ymin, true_xmin_ymin)
	intersect_xmax_ymax = tf.minimum(pred_xmax_ymax, true_xmax_ymax)
	intersect_wh = tf.maximum(intersect_xmax_ymax - intersect_xmin_ymin, 0)

	intersect_area = intersect_wh[...,0] * intersect_wh[..., 1]
	union_area = pred_wh[...,0]*pred_wh[...,1] + true_wh[...,0]*true_wh[...,1] - intersect_area

	iou = intersect_area / (union_area + 1e-6)

	union_xmin_ymin = tf.minimum(pred_xmin_ymin, true_xmin_ymin)
	union_xmax_ymax = tf.maximum(pred_xmax_ymax, true_xmax_ymax)
	union_wh = tf.maximum(union_xmax_ymax-union_xmin_ymin, 0)
	enclosed_area = union_wh[...,0] * union_wh[...,1]
	# iou = tf.Print(iou, [iou], message="debug iou")

	diou = iou - tf.reduce_sum(tf.square(pred_xy-true_xy), axis=-1)\
	       /(tf.reduce_sum(tf.square(union_xmax_ymax-union_xmin_ymin), axis=-1) + 1e-6)
	m = tf.square(tf.atan(true_wh[...,0]/(true_wh[...,1]+1e-6)) - tf.atan(pred_wh[...,0]/(pred_wh[..., 1]+1e-6)))
	# m = tf.Print(m, [m], message="debug m")
	v = tf.square(tf.constant(2.0/math.pi, dtype=true_wh[...,0].dtype)) * m
	# v = tf.Print(v, [v], message="debug V")
	alpha = v/(1-iou+v)

	giou = iou - (enclosed_area - union_area)/ (enclosed_area + 1e-6)
	giou = tf.Print(giou, [giou], message="debug giou")
	# alpha = tf.Print(alpha, [alpha], message="debug alpha")
	# diou = tf.Print(diou, [diou], message="debug diou")
	ciou = diou - alpha * v
	box_loss_scale = tf.squeeze(box_loss_scale, -1)
	ciou_loss = true_box[...,4]*box_loss_scale*(1-ciou)
	giou_loss = true_box[..., 4] * box_loss_scale * (1 - giou)
	iou_loss = true_box[..., 4] * box_loss_scale*(1-iou)

	return iou_loss


if __name__ == '__main__':

	gt_holder1 = tf.placeholder(tf.float32, (4, 10, 10, 3, 6))
	gt_holder2 = tf.placeholder(tf.float32, (4, 20, 20, 3, 6))
	gt_holder3 = tf.placeholder(tf.float32, (4, 40, 40, 3, 6))
	gt_holder = [gt_holder1, gt_holder2, gt_holder3]

	pred_holder1 = tf.placeholder(tf.float32, (4, 10, 10, 3, 6))
	pred_holder2 = tf.placeholder(tf.float32, (4, 20, 20, 3, 6))
	pred_holder3 = tf.placeholder(tf.float32, (4, 40, 40, 3, 6))
	pred_holder = [pred_holder1, pred_holder2, pred_holder3]

	anchors = np.array([(243, 146), (244, 140), (248, 154), (250, 146), (250, 149), (252, 143), (253, 145), (254, 148), (255, 150)])
	loss = yolo_loss(gt_holder, pred_holder, anchors)
