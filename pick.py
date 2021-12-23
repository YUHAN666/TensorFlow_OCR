import os
import cv2


source_root = 'C:/Users/yinha/source/repos/PAT_AOI/PAT_AOI/bin/x64/Debug/ng3'
image_root = 'E:/CODES/FAST-SCNN/DATA/carrier2'
dst_root = 'E:/DATA/NEW/'
image_names = [i[2] for i in os.walk(source_root)][0]

for i in image_names:
	name = i.split('.')[0]+'.bmp'
	src_path = os.path.join(image_root, name)

	dst_path = os.path.join(dst_root, i)
	try:
		image = cv2.imread(src_path, 0)
		cv2.imwrite(dst_path, image)
	except:
		continue
