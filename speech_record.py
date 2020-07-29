import os
import numpy as np
import time
import cv2
import dlib
from imutils import face_utils
import imutils
import sys
from threading import Timer
from queue import Queue

def update():
    timer = Timer(0.05,update)
    timer.start()
    q.put_nowait(capture.read()[1])

def save_video(frames,i):
    fourcc = cv2.VideoWriter_fourcc(*'I420')
    fps = 20
    size = (320,180)
    #command = "open"
    save_name = "./test/test"+str(i)+".avi"
    #save_name = f"/Users/soshiyuu/Devlopment/Github/zxsu_10words_dataset/{command}/{command}"+str(i)+".avi"
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)

    for frame in frames:
        video_writer.write(frame)
    video_writer.release()
    print(f"Saved {save_name}")

if __name__ == '__main__':
    
    
    global q
    global capture
    capture = cv2.VideoCapture(0)
    q = Queue(maxsize=100)

    update()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/Users/soshiyuu/Downloads/code/shape_predictor_68_face_landmarks.dat")
    buffer = Queue(maxsize=10)

    mo = False
    record = []
    i=0
    while True:
        if q.empty():
            time.sleep(0.01)
        else:
            frame = q.get()
            image = imutils.resize(frame, width=320)
            if buffer.full():
                buffer.get_nowait()
            buffer.put_nowait(image)
            rects = detector(image, 1)
            for (_, rect) in enumerate(rects):
                shape = predictor(image, rect)
                shape = face_utils.shape_to_np(shape)
            if mo:
                while not buffer.empty():
                        record.append(buffer.get())
            if rects:
                inside_rect = cv2.boundingRect(np.array([shape[61:68]]))
                if inside_rect[3]/inside_rect[2] >0.3:
                    if not mo:
                        print("capturing speech")
                    t1 = 0
                    mo = True
                if mo and inside_rect[3]/inside_rect[2] <0.3:
                    if not t1:
                        t1 = time.time()
                    elif time.time() - t1 > 0.5:
                        mo = False
                        print("speech finished")
                        save_video(record,i)
                        i+=1
                        record = []
    capture.release()


    