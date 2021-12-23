import cv2
import os
import numpy as np


image_root = "E:/123/"
image_names = [i[2] for i in os.walk(image_root)][0]

for path in image_names:
	image_path = os.path.join(image_root, path)
	image = cv2.imread(image_path, 0)
	img = cv2.GaussianBlur(image, (13,13), 3.0)


	# img = cv2.equalizeHist(image)
	# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
	# img = clahe.apply(image)
	# image = cv2.resize(image, (1280,640))
	# img = cv2.resize(img, (1280,640))
	# cv2.imshow("image", image)
	# cv2.imshow("img", img)
	# cv2.waitKey()
	cv2.imwrite(os.path.join("E:/save/", path), img)




