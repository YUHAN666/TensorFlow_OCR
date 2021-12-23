import cv2
import numpy as np
import tensorflow as tf


def db_cut_head(threshold_map, cluster_num):

	grid_shape = threshold_map.shape[1:3]
	# points_x = tf.where(threshold_map>0.3, grid_x, tf.zeros_like(grid_x))
	# points_y = tf.where(threshold_map>0.3, grid_y, tf.zeros_like(grid_y))
	# points = tf.concat([points_x, points_y], axis=-1)
	centers = []
	for b in range(threshold_map.shape[0]):
		grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1]), [grid_shape[0], 1, 1])
		grid_y = tf.tile(tf.reshape(tf.range(0, threshold_map.shape[1]), [-1, 1, 1]), [1, grid_shape[1], 1])
		grid_xy = tf.concat([grid_x, grid_y], axis=-1)
		mask = threshold_map[b,:,:,:] > 0.3
		mask = tf.squeeze(mask, -1)
		points = tf.boolean_mask(grid_xy, mask)

		centers.append(kmeans_tf(points, cluster_num))

	return centers


def kmeans_tf(points, cluster_num):
	"""
	:param points:  (N, 2)
	:param cluster_num:
	:return:
	"""
	centers = tf.gather(points,tf.random.uniform(shape=(cluster_num,), minval=0, maxval=tf.shape(points)[0]-1, dtype=points.dtype) )
	# centers = tf.Print(centers, [centers], 'debug centers')
	# centers_x = tf.random.uniform(shape=(cluster_num,1), minval=0, maxval=grid_shape[1]-1, dtype=points.dtype)
	# centers_y = tf.random.uniform(shape=(cluster_num,1), minval=0, maxval=grid_shape[0]-1, dtype=points.dtype)
	# centers = tf.concat([centers_x, centers_y], -1)     # (cluster_num, 2)

	centers = tf.expand_dims(centers, axis=0)       # (1, cluster_num, 2)
	points = tf.expand_dims(points, -2)             # (N, 1, 2)
	distance = tf.reduce_sum(tf.square(points - centers), axis=-1, keepdims=False)  # (N, cluster_num)
	max_index = tf.arg_min(distance, dimension=1)           # (N, 1)
	# center_points = tf.TensorArray(points.dtype, dynamic_size=True)
	#
	# def kmeans_loop_body(cluster_num, max_index, i, center_points):
	# 	mask = max_index == i
	# 	cluster = tf.boolean_mask(points, mask)
	# 	new_cluster = tf.reduce_mean(cluster, axis=0)
	# 	i += 1
	# 	center_points = center_points.write(i, new_cluster)
	#
	# 	return cluster_num, max_index, i, center_points
	#
	# def kmeans_loop_cond(cluster_num, max_index, i, center_points):
	#
	# 	return i<cluster_num
	#
	# _, _, _, center_points = tf.while_loop(kmeans_loop_cond, kmeans_loop_body, loop_vars=[cluster_num, max_index, 0, center_points])

	new_centers = tf.TensorArray(points.dtype, size=cluster_num, dynamic_size=False,)
	old_centers = tf.TensorArray(points.dtype, size=cluster_num, dynamic_size=False,)
	for c in range(cluster_num):
		mask = tf.equal(max_index,c)
		cluster = tf.boolean_mask(tf.squeeze(points, -2), mask)
		new_cluster = tf.reduce_mean(cluster, axis=0)
		new_centers = new_centers.write(c, new_cluster)
		old_centers = old_centers.write(c, centers[0, c, 0:])
	new_centers = new_centers.stack()
	old_centers = old_centers.stack()
	distance = tf.reduce_sum(tf.square(points - new_centers), axis=-1, keepdims=False)  # (N, cluster_num)
	min_index = tf.arg_min(distance, dimension=1)           # (N, 1)

	def kmeans_loop_body(old_centers, new_centers, min_index, points):

		old_centers = new_centers
		new_centers = tf.TensorArray(points.dtype, size=cluster_num, dynamic_size=False)
		for c in range(cluster_num):
			mask = tf.equal(min_index, c)
			cluster = tf.boolean_mask(tf.squeeze(points, -2), mask)
			new_cluster = tf.reduce_mean(cluster, axis=0)
			new_centers = new_centers.write(c, new_cluster)

		new_centers = new_centers.stack()
		distance = tf.reduce_sum(tf.square(points - new_centers), axis=-1, keepdims=False)  # (N, cluster_num)
		min_index = tf.arg_min(distance, dimension=1)  # (N, 1)
		# new_centers = tf.Print(new_centers, [new_centers], "Debug new_centers",summarize=4)

		return old_centers, new_centers, min_index, points


	def kmeans_loop_cond(old_centers, new_centers, *args):

		judge = tf.reduce_sum(tf.square(old_centers - new_centers), keepdims=False) > 0

		return judge

	_, kmeans_centers, _, _ = tf.while_loop(kmeans_loop_cond, kmeans_loop_body, loop_vars=[old_centers, new_centers, min_index, points])


	return kmeans_centers


if __name__ == '__main__':

	input_shape = (640, 640,3)
	image_map = np.ones(input_shape, dtype=np.float32)*255.0
	num_points = 10000
	cluster_num = 4

	pointsx = np.random.randint(0, input_shape[1]//3-1, num_points)
	pointsy = np.random.randint(0, input_shape[0]//3-1, num_points)
	p1 = np.concatenate([np.reshape(pointsx,(-1, 1)), np.reshape(pointsy, (-1,1))], axis=-1)
	points = [(pointsx[i], pointsy[i]) for i in range(num_points)]
	# cv2.drawKeypoints(image_map, (pointsx, pointsy), image_map, (0,0,255))
	for i in range(num_points):
		cv2.drawMarker(image_map, points[i], (0,255,0), 1, 2, 1)

	pointsx = np.random.randint(input_shape[1]*2//3, input_shape[1]-1, num_points)
	pointsy = np.random.randint(input_shape[0]*2//3, input_shape[0]-1, num_points)
	p2 = np.concatenate([np.reshape(pointsx,(-1, 1)), np.reshape(pointsy, (-1,1))], axis=-1)
	points = [(pointsx[i], pointsy[i]) for i in range(num_points)]
	# cv2.drawKeypoints(image_map, (pointsx, pointsy), image_map, (0,0,255))
	for i in range(num_points):
		cv2.drawMarker(image_map, points[i], (0,255,0), 1, 2, 1)
	#
	pointsx = np.random.randint(0, input_shape[1]//3-1, num_points)
	pointsy = np.random.randint(input_shape[0]*2//3, input_shape[0]-1, num_points)
	p3 = np.concatenate([np.reshape(pointsx,(-1, 1)), np.reshape(pointsy, (-1,1))], axis=-1)
	points = [(pointsx[i], pointsy[i]) for i in range(num_points)]
	# cv2.drawKeypoints(image_map, (pointsx, pointsy), image_map, (0,0,255))
	for i in range(num_points):
		cv2.drawMarker(image_map, points[i], (0,255,0), 1, 2, 1)

	pointsx = np.random.randint(input_shape[1]*2//3, input_shape[1]-1, num_points)
	pointsy = np.random.randint(0, input_shape[0]//3-1, num_points)
	p4 = np.concatenate([np.reshape(pointsx,(-1, 1)), np.reshape(pointsy, (-1,1))], axis=-1)
	points = [(pointsx[i], pointsy[i]) for i in range(num_points)]
	# cv2.drawKeypoints(image_map, (pointsx, pointsy), image_map, (0,0,255))
	for i in range(num_points):
		cv2.drawMarker(image_map, points[i], (0,255,0), 1, 2, 1)

	p = np.concatenate([p1, p2, p3, p4], 0)
	# p = np.concatenate([p1, p2], 0)
	# p = p1

	points_placeholder = tf.placeholder(tf.int32, p.shape)

	sess = tf.Session()

	with sess.as_default():
		kmeans_output= kmeans_tf(points_placeholder, cluster_num)

	for j in range(10):

		kmeans_centers = sess.run(kmeans_output, feed_dict={points_placeholder: p})
		image = image_map.copy()
		for i in range(cluster_num):
			cv2.drawMarker(image, tuple(kmeans_centers[0][i]), (0, 0, 255), 1, 20, 1)
		cv2.imshow('image', image)
		cv2.waitKey()
		cv2.destroyAllWindows()



