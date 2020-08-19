import os
import cv2
import time
import numpy as np
import face_recognition
import matplotlib.pyplot as plt

path = './datasets/single'
crop_path = 'datasets/cropped'
images, classNames = [], []
myList = os.listdir(path)
count = 0
for dir in myList:
    dir_name = path + '/' + dir
    if not os.path.isdir(crop_path + '/' + dir):
        os.mkdir(crop_path + '/' + dir)
    if '.DS_Store' not in dir:
        for filename in os.listdir(dir_name):
            if filename != '.DS_Store':
                # curImg = face_recognition.load_image_file(filename)
                curImg = cv2.imread(dir_name + '/' + filename)
                top, right, bottom, left = face_recognition.face_locations(curImg)[
                    0]
                # print('top, right, bottom, left: ', top, right, bottom, left)
                img_crop = curImg[top:bottom, left:right]
                img_crop = cv2.resize(img_crop, (128, 128))
                # curImg = cv2.rectangle(curImg, (left, top), (right, bottom), (0, 255, 0), 2)
                # print(filename)
                print(crop_path + '/' + dir + '/' + filename, img_crop.shape)
                cv2.imwrite(crop_path + '/' + dir + '/' + filename, img_crop)
                count += 1

print('\nNumber of classes: %d\n' % count)
