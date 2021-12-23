import cv2
import numpy as np


class YOLO_Kmeans:

	def __init__(self, cluster_number, txt_filename, anchor_filename):
		self.cluster_number = cluster_number
		self.txt_filename = txt_filename
		self.anchor_filename = anchor_filename

	def iou(self, boxes, clusters):  # 1 box -> k clusters
		n = boxes.shape[0]
		k = self.cluster_number

		box_area = boxes[:, 0] * boxes[:, 1]
		box_area = box_area.repeat(k)
		box_area = np.reshape(box_area, (n, k))

		cluster_area = clusters[:, 0] * clusters[:, 1]
		cluster_area = np.tile(cluster_area, [1, n])
		cluster_area = np.reshape(cluster_area, (n, k))

		box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
		cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
		min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

		box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
		cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
		min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
		inter_area = np.multiply(min_w_matrix, min_h_matrix)

		result = inter_area / (box_area + cluster_area - inter_area + 1e-4)
		return result

	def avg_iou(self, boxes, clusters):
		accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
		return accuracy

	def kmeans(self, boxes, k, dist=np.median):
		box_number = boxes.shape[0]
		distances = np.empty((box_number, k))
		last_nearest = np.zeros((box_number,))
		clusters = boxes[np.random.choice(
			box_number, k, replace=False)]  # init k clusters
		while True:

			distances = 1 - self.iou(boxes, clusters)

			current_nearest = np.argmin(distances, axis=1)
			if (last_nearest == current_nearest).all():
				break  # clusters won't change
			for cluster in range(k):
				clusters[cluster] = dist(  # update clusters
					boxes[current_nearest == cluster], axis=0)

			last_nearest = current_nearest

		return clusters

	def result2txt(self, data):
		f = open(self.anchor_filename, 'w')
		row = np.shape(data)[0]
		for i in range(row):
			if i == 0:
				x_y = "%d,%d" % (data[i][0], data[i][1])
			else:
				x_y = ", %d,%d" % (data[i][0], data[i][1])
			f.write(x_y)
		f.close()

	def txt2boxes(self):
		f = open(self.txt_filename, 'r')
		dataSet = []
		for line in f:
			infos = line.split(" ")
			length = len(infos)
			for i in range(1, length):
				width = int(infos[i].split(",")[2]) - \
				        int(infos[i].split(",")[0])
				height = int(infos[i].split(",")[3]) - \
				         int(infos[i].split(",")[1])
				dataSet.append([width, height])
		result = np.array(dataSet)
		f.close()
		return result

	def txt2clusters(self):
		all_boxes = self.txt2boxes()
		result = self.kmeans(all_boxes, k=self.cluster_number)
		result = result[np.lexsort(result.T[0, None])]
		self.result2txt(result)
		print("K anchors:\n {}".format(result))
		print("Accuracy: {:.2f}%".format(
			self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":

	if __name__ == '__main__':

		input_shape = (640, 640, 3)
		image_map = np.ones(input_shape, dtype=np.float32) * 255.0
		num_points = 10000
		cluster_num = 2

		pointsx = np.random.randint(0, input_shape[1] // 4 - 1, num_points)
		pointsy = np.random.randint(0, input_shape[0] // 4 - 1, num_points)
		p1 = np.concatenate([np.reshape(pointsx, (-1, 1)), np.reshape(pointsy, (-1, 1))], axis=-1)
		points = [(pointsx[i], pointsy[i]) for i in range(num_points)]
		# cv2.drawKeypoints(image_map, (pointsx, pointsy), image_map, (0,0,255))
		for i in range(num_points):
			cv2.drawMarker(image_map, points[i], (0, 255, 0), 1, 2, 1)

		pointsx = np.random.randint(input_shape[1] * 3 // 4, input_shape[1] - 1, num_points)
		pointsy = np.random.randint(input_shape[0] * 3 // 4, input_shape[0] - 1, num_points)
		p2 = np.concatenate([np.reshape(pointsx, (-1, 1)), np.reshape(pointsy, (-1, 1))], axis=-1)
		points = [(pointsx[i], pointsy[i]) for i in range(num_points)]
		# cv2.drawKeypoints(image_map, (pointsx, pointsy), image_map, (0,0,255))
		for i in range(num_points):
			cv2.drawMarker(image_map, points[i], (0, 255, 0), 1, 2, 1)
		#
		pointsx = np.random.randint(0, input_shape[1] // 4 - 1, num_points)
		pointsy = np.random.randint(input_shape[0] * 3 // 4, input_shape[0] - 1, num_points)
		p3 = np.concatenate([np.reshape(pointsx, (-1, 1)), np.reshape(pointsy, (-1, 1))], axis=-1)
		points = [(pointsx[i], pointsy[i]) for i in range(num_points)]
		# cv2.drawKeypoints(image_map, (pointsx, pointsy), image_map, (0,0,255))
		for i in range(num_points):
			cv2.drawMarker(image_map, points[i], (0, 255, 0), 1, 2, 1)

		pointsx = np.random.randint(input_shape[1] * 3 // 4, input_shape[1] - 1, num_points)
		pointsy = np.random.randint(0, input_shape[0] // 4 - 1, num_points)
		p4 = np.concatenate([np.reshape(pointsx, (-1, 1)), np.reshape(pointsy, (-1, 1))], axis=-1)
		points = [(pointsx[i], pointsy[i]) for i in range(num_points)]
		# cv2.drawKeypoints(image_map, (pointsx, pointsy), image_map, (0,0,255))
		for i in range(num_points):
			cv2.drawMarker(image_map, points[i], (0, 255, 0), 1, 2, 1)

		p = np.concatenate([p1, p2, p3], 0)

		kmeans_manager = YOLO_Kmeans(3, None, None)
		cluster = kmeans_manager.kmeans(p, 3)

		for j in range(10):

			image = image_map.copy()
			for i in range(len(cluster)):
				cv2.drawMarker(image, tuple(cluster[i]), (0, 0, 255), 1, 20, 1)

			cv2.imshow('img', image)
			cv2.waitKey()
			cv2.destroyAllWindows()



