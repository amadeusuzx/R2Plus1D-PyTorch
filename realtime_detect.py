import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VideoDataset, VideoDataset1M
from network import R2Plus1DClassifier

import cv2
import dlib
from imutils import face_utils
import imutils
import sys
import threading
import queue

import tempfile
from multiprocessing import Process


class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue(maxsize=100)
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.q.put_nowait(frame)

    def read(self):
        return self.q.get()

def recognize(record,model,j):
    size = (86,34)
    
    lip = record[0][1]
    overall_h = int(lip[3]*3)*4
    overall_w = int(lip[2]*2)*4
    buffer = np.empty((len(record), size[1], size[0], 3), np.dtype('float32'))
    i=0
    # fourcc = cv2.VideoWriter_fourcc(*'I420')
    # fps = 30
    # size = (256,160)
    # save_name = f"/Users/soshiyuu/Devlopment/Github/zxsu_10words_dataset/test"+str(j)+".avi"
    # video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)
    for entry in record:
        lip = entry[1]
        frame = entry[0]
        center = (lip[0]*4+lip[2]*2,lip[1]*4+lip[3]*2)

        frame = cv2.resize(frame[center[1]-overall_h//2:center[1]+overall_h//2,center[0]-overall_w//2:center[0]+overall_w//2],size)
        buffer[i] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #video_writer.write(frame)
        i+=1


    #video_writer.release()
    sampling = np.linspace(0, len(record)-1, num=16, dtype=int)
    
    buffer = buffer[sampling]
    buffer = buffer.transpose((3, 0, 1, 2))
    buffer = (buffer - np.mean(buffer))/np.std(buffer)
    buffer = torch.tensor(buffer)
    buffer = buffer.reshape([1]+[s for s in buffer.shape])
    outputs = model(buffer)
    _,preds = torch.max(outputs, 1)
    #commands = sorted(["black","cancel","centeralign","copy","large","medium","newslide","paste","red","textbox"])
    commands = sorted(["open",
            "close",
            "press",
            "release",
            "scroll_up",
            "scroll_down",
            "task_switch",
    ])
    #print(outputs)
    for p in preds:
        print(commands[p])

if __name__ == "__main__":

    num_classes = 7
    layer_sizes=[2, 2, 2, 2]
    path="model_zxsu_std.model"

    model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=layer_sizes)

    checkpoint = torch.load(path,map_location=torch.device('cpu'))
    print("Reloading from checkpoint")
    # restores the model and optimizer state_dicts
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    global detector 
    global predictor
    cap = VideoCapture(0)


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/Users/soshiyuu/Downloads/code/shape_predictor_68_face_landmarks.dat")
    buffer = queue.Queue(maxsize=10)

    mo = False
    record = []
    j=0
    while True:
        if cap.q.empty():
            time.sleep(0.01)
        else:
            frame = cap.read()
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
                        if len(record)>20:
                            p = Process(target=recognize, args=(record,model,j,))
                            p.start()
                        j+=1
                        record = []
