import os
import time
import numpy as np


import cv2
import dlib
import imutils

from imutils import face_utils

import sys
from multiprocessing import Process
from queue import Queue

def recognize(record,j):
    lip = record[0][1]
    overall_h = int(lip[3]*3)*4
    overall_w = int(lip[2]*2)*4
    size = (256,160)
    buffer = np.empty((len(record), size[1],size[0], 3), np.dtype('float32'))

    i=0
    fourcc = cv2.VideoWriter_fourcc(*'I420')
    fps = 30
    
    save_name = f"/Users/soshiyuu/Devlopment/Github/zxsu_10words_dataset/256_160data/"+str(j)+".avi"
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)
    for entry in record:
        lip = entry[1]
        frame = entry[0]
        center = (lip[0]*4+lip[2]*2,lip[1]*4+lip[3]*2)

        frame = cv2.resize(frame[center[1]-overall_h//2:center[1]+overall_h//2,center[0]-overall_w//2:center[0]+overall_w//2],size)
        buffer[i] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
        i+=1


    video_writer.release()
    print(f"saved {save_name}")

if __name__ == "__main__":


    global detector 
    global predictor
    
    capture = cv2.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/Users/soshiyuu/Downloads/code/shape_predictor_68_face_landmarks.dat")

    buffer = Queue(maxsize=15)
    mo = False
    record = []
    j=0
    ret, img = capture.read()
    print("intialized ",ret)

    while ret:
        ret,frame = capture.read()
        image = cv2.cvtColor(cv2.resize(frame, (320,180)),cv2.COLOR_BGR2GRAY)
        if buffer.full():
            buffer.get_nowait()
        
        rects = detector(image, 1)
        for (_, rect) in enumerate(rects):
            shape = predictor(image, rect)
            shape = face_utils.shape_to_np(shape)
        if mo:
            while not buffer.empty():
                record.append(buffer.get())
        if rects:
            lip = cv2.boundingRect(np.array([shape[48:68]]))
            inside_rect = cv2.boundingRect(np.array([shape[61:68]]))
            buffer.put_nowait([frame,lip])
            if inside_rect[3]/inside_rect[2] >0.4:
                if not mo:
                    print("capturing speech")
                t1 = 0
                mo = True
            if mo and inside_rect[3]/inside_rect[2] <0.4:
                if not t1:
                    t1 = time.time()
                elif time.time() - t1 > 1:
                    mo = False
                    print("speech finished")
                    p = Process(target=recognize, args=(record,j,))
                    p.start()
                    j+=1
                    record = []
