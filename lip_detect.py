import os
import numpy as np
import time
import cv2
import dlib
from imutils import face_utils
import imutils
import sys


if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/Users/soshiyuu/Downloads/code/shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(0)
    print("initialized")
    lip = False
    point_color = (0, 0, 255)
    thickness = 4
    lineType = 4
    mo = False
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            image = imutils.resize(frame, width=320)
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(image, 1)
            for (_, rect) in enumerate(rects):
                shape = predictor(image, rect)
                shape = face_utils.shape_to_np(shape)
                lip = cv2.boundingRect(np.array([shape[48:60]]))
            if lip:
                inside_rect = cv2.boundingRect(np.array([shape[61:68]]))
                if inside_rect[3]/inside_rect[2] >0.4:
                    print("capture speech")
                    t1 = 0
                    mo = True
                if mo and inside_rect[3]/inside_rect[2] <0.25:
                    if not t1:
                        t1 = time.time()
                    elif time.time() - t1 > 0.5:
                        mo = False
                        print("speech finished")

                roi = image[lip[1]-lip[3]//2:lip[1] + 3*lip[3]//2, lip[0] - lip[2]//2:lip[0] + 3*lip[2]//2]
                roi = cv2.resize(roi,(80,60))
                cv2.imshow('image', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    cv2.destroyAllWindows()
    sys.exit(0)
    
    