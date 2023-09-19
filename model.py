import cv2
import numpy as np
import os
import time
from matplotlib import pyplot as ply
import uuid
import torch


cap =cv2.VideoCapture(0)
model=torch.hub.load('ultralytics/yolov5','custom',path='E:/ODETECT/drowsiness/yolov5/yolov5-master/runs/train/exp9/weights/best.pt',force_reload=True)
while cap.isOpened():
    ret,frame=cap.read()
    results = model(frame)
    cv2.imshow('Model',np.squeeze(results.render()))
    
    if cv2.waitKey(30)& 0xFF== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()