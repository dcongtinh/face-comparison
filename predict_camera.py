import os
import cv2
import ssl
import time
import urllib.request
import numpy as np
import face_recognition
from scipy import spatial
import matplotlib.pyplot as plt
from PIL import ImageGrab

file_encodings = 'encodings_aug.npy'
file_names = 'names.npy'
threshold = 0.95


def importFaceEncoded(filename):
    # with open(filename, 'rb') as f:
    encodeListKnown = np.load(filename)
    print(filename + ' has imported!')
    return encodeListKnown


encodings = importFaceEncoded(file_encodings)
names = importFaceEncoded(file_names)

dateStr = time.strftime('%Y-%m-%d')
attend_dir = 'Attendence/' + dateStr
if not os.path.isdir(attend_dir):
    os.mkdir(attend_dir)

attend_filename = attend_dir + '/Attendance_%s.csv' % dateStr
if not os.path.isfile(attend_filename):
    with open(attend_filename, 'w+') as f:
        f.writelines('mssv,confidence,date,time\n')

with open(attend_filename, 'r') as f:
    myDataList = f.readlines()
    nameList = []
    for line in myDataList:
        entry = line.split(',')
        nameList.append(entry[0])


# FOR CAPTURING SCREEN RATHER THAN WEBCAM
def captureScreen(name, bbox=(300, 300, 690+300, 530+300)):
    capScr = np.array(ImageGrab.grab(bbox))
    capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name + '.jpg', capScr)


def markAttendance(name, confidence):
    with open(attend_filename, 'a+') as f:
        if name not in nameList:
            timeStr = time.strftime('%H:%M:%S')
            f.writelines('%s,%.6f,%s,%s\n' %
                         (name, confidence, dateStr, timeStr))
            nameList.append(name)
            # captureScreen(name)


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    arr = []
    for face_encoding in face_encodings:
        arr.append(spatial.distance.cosine(face_encoding, face_to_compare))

    return np.array(arr)


def draw_box(img, face_encodings, face_locations):
    fontFace = cv2.FONT_HERSHEY_COMPLEX
    count, fontScale, thickness = 0, 0.5, 1
    red, green = (0, 0, 255), (0, 255, 0)
    for encodeFace, faceLoc in zip(face_encodings, face_locations):
        tolerance = 0.6
        faceDis = face_distance(encodings, encodeFace)
        matchIndex = np.argmin(faceDis)
        confidence = 1 - faceDis[matchIndex]
        if faceDis[matchIndex] <= tolerance:
            y1, x2, y2, x1 = faceLoc
            color = green
            if confidence >= threshold:
                name = names[matchIndex].upper()
                label = '%s %.4f' % (name, confidence)
                markAttendance(name, confidence)
                cv2.imwrite('%s/%s_%s.jpg' % (attend_dir, name, dateStr),
                            cv2.resize(img[y1:y2, x1:x2], (128, 128)))  # save image
                count += 1
            else:
                label = 'Unknown~'
                color = red

            labelSize = cv2.getTextSize(
                label, fontFace, fontScale, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            cv2.rectangle(
                img, (x1, y1), (x1+labelSize[0], y1-labelSize[1]-20), color, cv2.FILLED)
            cv2.putText(img, label, (x1, y1-10),
                        fontFace, fontScale, (0, 0, 0), thickness)
    cnt_x1, cnt_y1 = 12, 12
    count_students = '%d student(s)' % count
    countSize = cv2.getTextSize(
        count_students, fontFace, fontScale, thickness)[0]
    cv2.rectangle(img, (cnt_x1, cnt_y1), (cnt_y1+countSize[0]+18, cnt_y1+countSize[1]+18),
                  green, -1)
    cv2.putText(img, count_students, (cnt_x1+8, countSize[1]+cnt_y1+8), fontFace,
                fontScale=fontScale, color=(0, 0, 0), thickness=thickness)
    return img


url = 'http://192.168.10.112:8080/shot.jpg'
print("[INFO] starting video stream...")
# cap = cv2.VideoCapture('http://192.168.10.83:8080/video')
# if (cap.isOpened() == False):
#     print("Error opening video stream or file")
# cv2.startWindowThread()
scale = 0.5

while True:
    # success, img = cap.read()
    # ---- Use urllib ----
    imgResp = urllib.request.urlopen(url)
    # Numpy to convert into a array
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    # Finally decode the array to OpenCV usable format ;)
    img = cv2.imdecode(imgNp, -1)
    # ---- End Use urllib ----
    img = cv2.resize(img, (0, 0), None, scale, scale)

    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(imgS)
    face_encodings = face_recognition.face_encodings(imgS, face_locations)

    img = draw_box(img, face_encodings, face_locations)
    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
