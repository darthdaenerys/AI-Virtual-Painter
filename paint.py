import cv2
import numpy as np
import os
from handTracker import findDistances,findError,MediapipeHands
import json
import pickle
import time
import sys

f=open('settings.json')
settings=json.load(f)
del f

drawState='Standby'
color='white'
brush_size=20

camera=cv2.VideoCapture(settings['camera_port'],cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,settings['window_height'])
camera.set(cv2.CAP_PROP_FRAME_WIDTH,settings['window_width'])
camera.set(cv2.CAP_PROP_FPS,settings['fps'])
camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

cv2.namedWindow('OpenCV Paint',cv2.WINDOW_NORMAL)
if settings['fullscreen']:
    cv2.setWindowProperty('OpenCV Paint',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

gesturenames=[]
knowngestures=[]
prevcanvas=np.zeros([settings['window_height'],settings['window_width'],3],dtype=np.uint8)
fps=0
fpsfilter=settings['fpsfilter']
starttime=time.time()
savetime=-1
run=True

if os.path.exists('gesture_data.pkl'):
    with open('gesture_data.pkl','rb') as f:
        gesturenames=pickle.load(f)
        knowngestures=pickle.load(f)
else:
    print('No gesture data found')
    sys.exit()

findhands=MediapipeHands(
    model_complexity=settings['model_complexity'],
    min_detection_confidence=settings['min_detection_confidence'],
    min_tracking_confidence=settings['min_tracking_confidence']
)
threshold=settings['confidence']
keypoints=settings['keypoints']
color_idx=['red','orange','yellow','green','cyan','blue','purple','pink','white','black']

def convert_toBNW(frame):
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    objectFrame=np.zeros([settings['window_height'],settings['window_width'],3],dtype=np.uint8)
    frame=cv2.addWeighted(frame,.8,objectFrame,.9,0)
    return frame

def clearcanvas():
    global prevcanvas
    prevcanvas=np.zeros([settings['window_height'],settings['window_width'],3],dtype=np.uint8)
