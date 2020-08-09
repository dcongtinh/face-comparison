import os
import cv2
import time
import numpy as np
import face_recognition
import matplotlib.pyplot as plt

path = './datasets/single'
images, classNames = [], []
myList = os.listdir(path)
count = 0
for cl in myList:
    filename = path + '/' + cl
    if '.DS_Store' not in filename:
        # curImg = face_recognition.load_image_file(filename)
        curImg = cv2.imread(filename)
        top, right, bottom, left = face_recognition.face_locations(curImg)[0]
        print('top, right, bottom, left: ', top, right, bottom, left)
        img_crop = curImg[top:bottom, left:right]
        img_crop = cv2.resize(img_crop, (96, 96))
        # curImg = cv2.rectangle(curImg, (left, top), (right, bottom), (0, 255, 0), 2)
        print(os.path.splitext(cl)[0] + '.jpg')
        cv2.imwrite('./datasets/face_cropped/' + os.path.splitext(cl)[0] + '.jpg', img_crop)
        count += 1

print('\nNumber of classes: %d\n' % count)
