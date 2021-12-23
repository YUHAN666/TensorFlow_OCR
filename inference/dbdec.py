import os
from timeit import default_timer as timer
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from dec.label_dict import num2label_dic
from utiles.box_utiles import polygons_from_bitmap
from utiles.transform import resize_image

IMAGE_SIZE = [320, 320]
CUT_SIZE = (64, 64)
db_model_path_1 = '../pbMode/chip_ocr_6_db_model_2.pb'
dec_model_path = '../pbMode/dec_model.pb'
image_root = 'D:/8/black_cut/'
# image_root = 'E:/CODES/FAST-SCNN/DATA/carrier2/'
sess = tf.Session()
with gfile.FastGFile(db_model_path_1, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
with gfile.FastGFile(dec_model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())

db_input_image_1 = sess.graph.get_tensor_by_name('step2/image_input:0')
db_out_1 = sess.graph.get_tensor_by_name('step2/dbnet/proba3_sigmoid:0')
dec_input_image = sess.graph.get_tensor_by_name('cut_image_input:0')
decision_out = sess.graph.get_tensor_by_name('decision_out:0')
image_names = [i[2] for i in os.walk(image_root)][0]
name_dic = {}

for i in image_names[0:1]:
    start = timer()
    image_path = os.path.join(image_root, i)

    image = cv2.imread(image_path)
    img = image.copy()
    image = image.astype(np.float32)
    image = resize_image(IMAGE_SIZE[0], image)
    image = image / 255.0
    # cv2.imshow('image', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    image = np.array(image[np.newaxis, :, :, :])
    proba_map_1 = sess.run(db_out_1, feed_dict={db_input_image_1: image})
    proba_map_1 = np.squeeze(proba_map_1, 0)
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    proba_map_1 = cv2.erode(proba_map_1, kernel_1)
    # kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    proba_map_1 = cv2.dilate(proba_map_1, kernel_1)
    bitmap = proba_map_1 > 0.5
    boxes, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image = np.squeeze(image, 0)
    # xwidth = 180
    # ywidth = 220
    xwidth = 15
    ywidth = 20
    scale = img.shape[1] / IMAGE_SIZE[0]
    if len(boxes) != 10:
        for j in range(len(boxes)):
            cv2.drawContours(image, boxes, j, (0,0,255))
        cv2.imshow('image', image)
        cv2.waitKey()
        continue
    for contour in boxes:
        contour = np.reshape(contour, (-1, 2))
        xmin = sorted(list(contour), key=lambda x: x[0])[0][0]*scale
        xmax = sorted(list(contour), key=lambda x: x[0])[-1][0]*scale
        ymin = sorted(list(contour), key=lambda x: x[1])[0][1]*scale
        ymax = sorted(list(contour), key=lambda x: x[1])[-1][1]*scale
        xcenter = int((xmin+xmax)//2)
        ycenter = int((ymin+ymax)//2)
        xmin = max(xcenter - xwidth, 0)
        ymin = max(ycenter - ywidth, 0)
        ymax = min(ycenter + ywidth, img.shape[0])
        xmax = min(xcenter + xwidth, img.shape[1])
        cut_image = img[ymin:ymax, xmin:xmax, :]
        _, mask = cv2.threshold(cut_image, 50, 1, cv2.THRESH_BINARY_INV)
        c, _ = cv2.findContours((mask[:,:,0] * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        thres = np.sum(mask)
        cut = cut_image.copy()
        qingxidu = cv2.Laplacian(cut, cv2.CV_32FC3).var()

        for j in range(len(c)):
            cv2.drawContours(cut, c, j, (0,0,255))
        cut = cv2.resize(cut, (128, 128))
        # cv2.putText(cut, str((int(qingxidu * 100) / 100)), (10, 20), 1, 1, (0, 0, 255))
        cv2.imshow('cut_image', cut)
        cv2.waitKey()
        # cv2.imshow('cut_image', cut_image)
        # cv2.waitKey()
        cut_image = cv2.resize(cut_image, CUT_SIZE)

        # cut_image = resize_image(CUT_SIZE[0], cut_image)
        cut_image = cut_image / 255.0
        cut_image = np.array(cut_image[np.newaxis, :, :, :])

        decision = sess.run(decision_out, feed_dict={dec_input_image: cut_image})[0]
        label = num2label_dic[str(decision)]
        # name_str.append((xmin, label))
        if label not in name_dic.keys():
            name_dic[label] = 1
        else:
            name_dic[label] += 1
        cut_image = np.squeeze(cut_image, 0)
        # cv2.imwrite(os.path.join('../cut/', label + '-' + str(name_dic[label]) + '.jpg'), cut_image*255.0)
    end = timer()
    print("time: {}".format(end-start))