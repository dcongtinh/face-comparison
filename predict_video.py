from datetime import datetime
import os
import cv2
import numpy as np
import face_recognition
from scipy import spatial
import matplotlib.pyplot as plt

file_encodings = 'encodings_aug.npy'
file_names = 'names.npy'
threshold = 0.95
# from PIL import ImageGrab


def importFaceEncoded(filename):
    # with open(filename, 'rb') as f:
    encodeListKnown = np.load(filename)
    print(filename + ' has imported!')
    return encodeListKnown


encodings = importFaceEncoded(file_encodings)
names = importFaceEncoded(file_names)

# def markAttendance(name):
#     with open('Attendance.csv', 'r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dtString}')

# # FOR CAPTURING SCREEN RATHER THAN WEBCAM
# # def captureScreen(bbox=(300,300,690+300,530+300)):
# #     capScr = np.array(ImageGrab.grab(bbox))
# #     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
# #     return capScr

def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    # normed = np.linalg.norm(face_encodings - face_to_compare, axis=1)
    # return normed.T * normed
    #----
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
            color = green
            if confidence >= threshold:
                name = '%s %.4f' % (
                    names[matchIndex].upper()[:8], confidence)
                count += 1
            else:
                name = 'Unknown~'
                color = red
            y1, x2, y2, x1 = faceLoc
            labelSize = cv2.getTextSize(
                name, fontFace, fontScale, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            cv2.rectangle(
                img, (x1, y1), (x1+labelSize[0], y1-labelSize[1]-20), color, cv2.FILLED)
            cv2.putText(img, name, (x1, y1-10),
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

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
scale = 0.5
while True:
    success, img = cap.read()
    #imgS = captureScreen()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (0, 0), None, scale, scale)

    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(imgS)
    face_encodings = face_recognition.face_encodings(imgS, face_locations)

    img = draw_box(img, face_encodings, face_locations)
    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
