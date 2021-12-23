import imgaug.augmenters as iaa
import cv2
import numpy
import os

image_root = './dataset/selfmade3/cut/train/'
image_list = [i[2] for i in os.walk(image_root)][0]

if __name__ == '__main__':

    for i in image_list:
        image = cv2.imread(os.path.join(image_root, i))
        transform = iaa.Sequential([iaa.GammaContrast((0.8, 1.2)),
                                                iaa.Affine(rotate=(-10, 10)),
                                                # iaa.Rotate(rotate=(-180, 180)),
                                                iaa.MotionBlur(k=(3, 7), angle=90, direction=(0, 0)),
                                                iaa.MotionBlur(k=(3, 7), angle=0, direction=(0, 0)),
                                                iaa.Crop(percent=(0, 0.1))], random_order=True)
        img = transform.augment_image(image)
        cv2.imshow('before augment', image)
        cv2.imshow('after augment', img)
        cv2.waitKey(0)

