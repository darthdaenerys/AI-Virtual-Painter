import cv2
import os
import pickle
import json
import sys

width=1280
height=720

choice=input('Load pretrained file? (Y/n): ')
gesturenames=[]
knowngestures=[]
run=True
train=True
cntr=0

f=open('settings.json')
data=json.load(f)

if choice.lower()=='y':
    if os.path.exists('gesture_data.pkl'):
        with open('gesture_data.pkl','rb') as f:
            gesturenames=pickle.load(f)
            knowngestures=pickle.load(f)
            train=False
    else:
        print('File does not exists!')
        sys.exit()
else:
    n=int(input('Enter number of gestures to train: '))
    for _ in range(n):
        x=input(f'Name of gesture {_+1}: ')
        gesturenames.append(x)

camera=cv2.VideoCapture(0,cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,width)
camera.set(cv2.CAP_PROP_FPS,30)
camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

cv2.namedWindow('Gesture Recognition')