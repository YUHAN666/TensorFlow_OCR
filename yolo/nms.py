import tensorflow as tf

from yolo.get_ious import get_iou_with_coordinates


def nms(pred_boxes, pred_scores, iou_threshold, max_boxes):

	# NMS with different iou strategy
	def nms_loop_cond(pred_boxes, pred_scores, nms_boxes, nms_scores, iou_threshold, i, max_boxes):
		return tf.logical_and(i < max_boxes, tf.shape(pred_boxes)[0] > 0)

	def nms_loop_body(pred_boxes, pred_scores, nms_boxes, nms_scores, iou_threshold, i, max_boxes):
		max_idx = tf.arg_max(pred_scores, dimension=0)
		max_box = tf.gather(pred_boxes, max_idx)
		max_score = tf.gather(pred_scores, max_idx)
		nms_boxes = nms_boxes.write(i, max_box)
		nms_scores = nms_scores.write(i, max_score)
		mask = (pred_scores < max_score)[..., 0]
		i += 1
		# i = i + 1
		# i = tf.Print(i,[i],'debug:i:')
		# pred_boxes = tf.concat([pred_boxes[:max_idx, ...], pred_boxes[max_idx + 1:, ...]], axis=0)
		# pred_scores = tf.concat([pred_scores[:max_idx,...], pred_scores[max_idx + 1:,...]], axis=0)
		pred_boxes = tf.boolean_mask(pred_boxes, mask)
		pred_scores = tf.boolean_mask(pred_scores, mask)

		ious = get_iou_with_coordinates(pred_boxes, max_box)
		# ious = tf.Print(ious, [ious], 'debug ious: ', summarize=1000)

		iou_mask = (ious < iou_threshold)[..., 0]

		pred_boxes = tf.boolean_mask(pred_boxes, iou_mask)
		pred_scores = tf.boolean_mask(pred_scores, iou_mask)

		return pred_boxes, pred_scores, nms_boxes, nms_scores, iou_threshold, i, max_boxes

	nms_scores = tf.TensorArray(dtype=pred_scores.dtype, size=1, dynamic_size=True)
	nms_boxes = tf.TensorArray(dtype=pred_boxes.dtype, size=1, dynamic_size=True)
	_, _, nms_boxes, nms_scores, _, _, _ = tf.while_loop(nms_loop_cond, nms_loop_body,
	                                                     [pred_boxes, pred_scores, nms_boxes, nms_scores,
	                                                      iou_threshold, 0, max_boxes])
	nms_scores = tf.squeeze(nms_scores.stack(), axis=-2)
	nms_boxes = tf.squeeze(nms_boxes.stack(), axis=-2)

	return nms_boxes, nms_scores