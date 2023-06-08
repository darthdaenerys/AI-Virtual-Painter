import cv2
import numpy as np
import json
import time

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