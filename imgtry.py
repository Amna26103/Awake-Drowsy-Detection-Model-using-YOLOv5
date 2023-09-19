import cv2
import numpy as np
import os
import time
from matplotlib import pyplot as ply
import uuid
import torch


model=torch.hub.load('ultralytics/yolov5','custom',path='E:/Object Detections/yolov5-master/runs/train/exp10/weights/best.pt',force_reload=True)
img="E:/Object Detections/data/images/d1.jpg"
results = model(img)
results.print()
