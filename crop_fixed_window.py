import cv2
import imutils
from imutils import face_utils
import imutils
import dlib
import numpy as np

def load_video(fname):
    capture = cv2.VideoCapture(fname)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
    buffer = []
    retaining = True
    count = 0
    # read in each frame, one at a time into the numpy buffer array
    while (count < frame_count and retaining):
        retaining, frame = capture.read()
        buffer.append(frame)
        count+=1
    capture.release()
    return buffer
if __name__ == "__main__":
    import os 
    from glob import glob
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/Users/soshiyuu/Downloads/code/shape_predictor_68_face_landmarks.dat")
    commands = ["open",
            "close",
            "press",
            "release",
            "scroll_up",
            "scroll_down",
            "task_switch",
    ]
    path = "./"
    save_path = "./lips/"

    for command in [commands]:
        os.makedirs(save_path+command)
        file_list = sorted(glob(path+command+"/*.avi"))
        i=0
        for f in file_list:
            i+=1
            frames = load_video(f)
            for frame in frames:
                rects = detector(frame, 1)
                for (_, rect) in enumerate(rects):
                    detected=True
                    shape = predictor(frame, rect)
                    shape = face_utils.shape_to_np(shape)
                    lip = cv2.boundingRect(np.array([shape[48:68]]))
                if detected :
                    break

            fourcc = cv2.VideoWriter_fourcc(*'I420')
            fps = 20
            size = (60,30)
            save_name = save_path+command+"/"+str(i)+".avi"
            video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)

            for frame in frames:
                frame = cv2.resize(frame[lip[1]-lip[3] : lip[1] + lip[3]*2, lip[0] - lip[2]//2:lip[0] + 3*lip[2]//2],(60,30))
                video_writer.write(frame)
            video_writer.release()
            print(f"Saved {save_name}")