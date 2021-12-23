import os

import cv2
import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf

from base.data_manager import DataManager
from dec.label_dict import *


class DataManagerDec(DataManager):

    def __init__(self, param):

        self.data_dir = param["data_dir"]
        self.image_size = param["cut_image_size"]
        self.batch_size = param["batch_size_cut"]
        self.batch_size_inference = param["batch_size_inference"]
        self.class_num = len(label2num_dic)

        # if not os.path.exists(os.path.join(self.data_dir, "cut/")):
        #     os.makedirs(os.path.join(self.data_dir, "cut", "train"))
        #     os.makedirs(os.path.join(self.data_dir, "cut", "test"))
        #     self.cut_image_from_gts()

        self.train_list, self.test_list = self.get_file_list()
        self.data_num_train = len(self.train_list)
        self.data_num_test = len(self.test_list)
        self.num_batch = self.data_num_train // self.batch_size

        self.dataset_train = tf.data.Dataset.from_generator(self.generator_train, (tf.float32, tf.float32))
        self.dataset_train = self.dataset_train.batch(self.batch_size).repeat()
        self.iterator_train = self.dataset_train.make_one_shot_iterator()
        self.next_batch_train = self.iterator_train.get_next()

        self.dataset_test = tf.data.Dataset.from_generator(self.generator_test, (tf.float32, tf.float32, tf.string))
        self.dataset_test = self.dataset_test.batch(self.batch_size_inference).repeat()
        self.iterator_test = self.dataset_test.make_one_shot_iterator()
        self.next_batch_test = self.iterator_test.get_next()

    def generator_train(self):

        current_index = 0
        index = np.arange(self.data_num_train)
        np.random.shuffle(index)
        while True:
            if current_index >= self.data_num_train:
                np.random.shuffle(index)
                current_index = 0

            image = cv2.imread(os.path.join(self.data_dir, 'train', self.train_list[index[current_index]]))
            image = image / 255.0
            image = cv2.resize(image, (self.image_size, self.image_size))

            label = label2num_dic[self.train_list[index[current_index]].split('-')[0]]
            label = self.scalar2onehot(label)
            current_index += 1
            if np.random.uniform() > 0.7:
                image_augment = iaa.Sequential([iaa.GammaContrast((0.8, 1.2)),
                                                # iaa.Rotate(rotate=(-180, 180)),
                                                iaa.MotionBlur(k=(3, 7), angle=90, direction=(0, 0)),
                                                iaa.MotionBlur(k=(3, 7), angle=0, direction=(0, 0)),
                                                iaa.Crop(percent=((0,0.02),(0,0.02), (0,0.02), (0,0.02)), sample_independently=True)], random_order=True)
                image = image_augment.augment_image(image)

            # cv2.imshow('image', image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            yield image, label

    def generator_test(self):

        current_index = 0
        index = np.arange(self.data_num_train)
        while True:
            if current_index >= self.data_num_train:
                current_index = 0
            path = os.path.join(self.data_dir, 'train', self.train_list[index[current_index]])
            image = cv2.imread(path)
            image = cv2.resize(image, (self.image_size, self.image_size))
            label = label2num_dic[self.train_list[index[current_index]].split('-')[0]]
            label = self.scalar2onehot(label)
            current_index += 1
            image = image/255.0



            yield image, label, path

    def get_file_list(self):

        train_list = [i[2] for i in os.walk(os.path.join(self.data_dir, 'train'))][0]
        test_list = [i[2] for i in os.walk(os.path.join(self.data_dir, 'test'))][0]

        return train_list, test_list

    def cut_image_from_gts(self):

        label_dict = {}
        with open(os.path.join(self.data_dir, "train_list.txt")) as f:

            for line in f.readlines():
                gts_name = os.path.join(self.data_dir, 'train_gts', line.rstrip('\n')) + '.txt'
                image_name = os.path.join(self.data_dir, 'train_images', line.rstrip('\n'))
                image = cv2.imread(image_name)
                with open(gts_name) as file:
                    for gt in file.readlines():
                        label = gt.rstrip("\n").split(',')[-1]
                        if label not in label_dict.keys():
                            label_dict[label] = 1
                        else:
                            label_dict[label] += 1
                        xmin = int(gt.split(',')[0])
                        xmax = int(gt.split(',')[2])
                        ymin = int(gt.split(',')[1])
                        ymax = int(gt.split(',')[5])

                        cut_image = image[ymin:ymax, xmin:xmax, :]
                        name = os.path.join(self.data_dir, "cut", "train", "{}-{}.jpg".format(label, str(label_dict[label])))
                        cv2.imwrite(name, cut_image)

        with open(os.path.join(self.data_dir, "test_list.txt")) as f:

            for line in f.readlines():
                gts_name = os.path.join(self.data_dir, 'test_gts', line.rstrip('\n')) + '.txt'
                image_name = os.path.join(self.data_dir, 'test_images', line.rstrip('\n'))
                image = cv2.imread(image_name)
                with open(gts_name) as file:
                    for gt in file.readlines():
                        label = gt.rstrip("\n").split(',')[-1]
                        if label not in label_dict.keys():
                            label_dict[label] = 1
                        else:
                            label_dict[label] += 1
                        xmin = int(gt.split(',')[0])
                        xmax = int(gt.split(',')[2])
                        ymin = int(gt.split(',')[1])
                        ymax = int(gt.split(',')[5])

                        cut_image = image[ymin:ymax, xmin:xmax, :]
                        name = os.path.join(self.data_dir, "cut", "test", "{}-{}.jpg".format(label, str(label_dict[label])))
                        cv2.imwrite(name, cut_image)

    def scalar2onehot(self, number):
        label = np.zeros((self.class_num,))
        label[number] = 1

        return label


if __name__ == "__main__":

    p = {"data_dir": "./dataset/selfmade3/", "cut_image_size": 32, 'batch_size_cut': 4, 'batch_size_inference':1}

    dm = DataManagerDec(p)

    session = tf.Session()
    data = session.run(dm.next_batch_train)

    cv2.imshow('image', data[0][0,:,:,:])
    cv2.waitKey()
    print(data)