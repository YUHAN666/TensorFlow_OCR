from timeit import default_timer as timer
import cv2
import numpy as np
from models.dbmodel_components.cut_head import db_cut_head
from utiles.box_utiles import polygons_from_bitmap


class Validator(object):
    def __init__(self, sess, model, logger, original_size):
        """ Validate the performance of model from checkpoint when training process is complete"""
        self.session = sess
        self.model = model
        self.logger = logger
        self.original_size = original_size
        with self.session.as_default():
            self.centers = db_cut_head(self.model.p, cluster_num=2)
            # self.cut_image = self.model.image_input[0, self.centers[0][0][0]-40:self.centers[0][0][0]+40, self.centers[0][0][1]-20:self.centers[0][0][1]+20, :]

    def visualization(self, data_manager):

        with self.session.as_default():
            # print('start validating segmentation')
            count = 0
            for batch in range(data_manager.num_batch_test):
                image_batch, _ = self.session.run(data_manager.next_batch_test)
                start = timer()
                proba_map = self.session.run([self.model.p], feed_dict={self.model.image_input: image_batch})
                end = timer()
                print('{}s'.format(end-start))
                proba_map = np.squeeze(proba_map[0], 0)
                image = np.squeeze(image_batch, 0)
                image *= 255.0
                bitmap = proba_map > 0.5
                boxes, scores = polygons_from_bitmap(proba_map, bitmap, image.shape[0], image.shape[1], box_thresh=0.5)
                # for box in boxes:
                #     cv2.drawContours(image, [np.array(box)], -1, (0, 255, 0), 2)
                #
                # # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                # # cv2.imshow('image', image)
                # # cv2.waitKey(0)
                # image = cv2.resize(image, self.original_size[::-1])
                # cv2.imwrite('./test/' + str(batch)+'.jpg', image)

                # contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                print(len(boxes))
                for contour in boxes:
                    contour = np.reshape(contour, (-1, 2))
                    xmin = sorted(list(contour), key=lambda x: x[0])[0][0]
                    xmax = sorted(list(contour), key=lambda x: x[0])[-1][0]
                    ymin = sorted(list(contour), key=lambda x: x[1])[0][1]
                    ymax = sorted(list(contour), key=lambda x: x[1])[-1][1]
                    # cut_image = image[ymin:ymax, xmin:xmax, :]
                    # cut_image = cv2.resize(cut_image, (96, 96))
                    # cv2.imwrite('./cut/' + str(batch)+'-'+str(count)+'.jpg', cut_image)
                    # count += 1
                    cv2.rectangle(image, (xmin, ymin), (xmax,ymax), (0,0,255))
                image = cv2.resize(image, (1024, 640))
                cv2.imshow('image', image/255.0)
                cv2.waitKey()
                # image = cv2.resize(image, self.original_size[::-1])
                # cv2.imwrite('./cut/' + str(batch) + '.jpg', image)


    def test_kmeans(self, data_manager, image_size):

        input_shape = data_manager.image_size
        scale = np.minimum(float(input_shape)/float(image_size[0]), float(input_shape)/float(image_size[1]))
        with self.session.as_default():
            # print('start validating segmentation')
            for batch in range(data_manager.num_batch_test):
                image_batch, original_image = self.session.run(data_manager.next_batch_test)
                start = timer()
                kmeans_centers = self.session.run(self.centers, feed_dict={self.model.image_input: image_batch})
                end = timer()
                print('total: {}s'.format(end - start))
                kmeans_centers = kmeans_centers / scale
                kmeans_centers = kmeans_centers.astype(np.int32)

                for b in range(data_manager.batch_size_inference):
                    centers = kmeans_centers[b]
                    image = original_image.copy() / 255.0
                    # image = image_batch.copy()[b,:,:,:]
                    for c in range(centers.shape[0]):
                        cv2.drawMarker(image, tuple(centers[c]), (0, 0, 255), 1, 20, 1)
                        img = image[centers[c][1] - 250:centers[c][1] + 250, centers[c][0] - 150:centers[c][0] + 150, :]
                        cv2.imshow("image", img)
                        cv2.waitKey()
                        cv2.destroyAllWindows()

                # cut_image = self.session.run(self.cut_image, feed_dict={self.model.image_input: image_batch})
                # cv2.imshow("image", cut_image)
                # cv2.waitKey()
