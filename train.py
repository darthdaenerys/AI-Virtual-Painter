import cv2
import os
import pickle
from handTracker import findDistances,findError,MediapipeHands
import json
import time
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

findhands=MediapipeHands()
starttime=-5
threshold=12

while train and cntr!=n:
    _,frame=camera.read()
    frame=cv2.flip(frame,1)
    frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    cv2.putText(frame,f'Press "s" to save gesture "{gesturenames[cntr]}"',(15,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    handlandmarks=findhands.handsdata(frameRGB)[0]
    if len(handlandmarks)==0:
        cv2.putText(frame,f'No hand detected!',(15,90),cv2.FONT_HERSHEY_SIMPLEX,1,(10,10,250),2)
    else:
        if cv2.waitKey(1) & 0xff==ord('s'):
            distMatrix=findDistances(handlandmarks[0])
            knowngestures.append(distMatrix)
            cntr+=1
            starttime=time.time()
    if time.time()-starttime<=1:
        cv2.putText(frame,f'Gesture saved succesfully',(width//2-400,height//2),cv2.FONT_HERSHEY_SIMPLEX,2,(10,250,10),2)
    cv2.imshow('Gesture Recognition',frame)
    if cv2.waitKey(1) & 0xff==ord('q'):
        train=False
        run=False

while run:
    _,frame=camera.read()
    frame=cv2.flip(frame,1)
    frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    handlandmarks=findhands.handsdata(frameRGB)
    if len(handlandmarks[0])>0:
        for hand in handlandmarks[0]:
            distMatrix=findDistances(hand)
            error,idx=findError(knowngestures,distMatrix, data['keypoints'])
            if error<threshold and idx!=-1:
                uppercoord=[9999,9999]
                for i in hand:
                    if i[1]<uppercoord[1]:
                        uppercoord=list(i)
                uppercoord[1]-=30
                uppercoord[0]-=30
                cv2.putText(frame,f'{gesturenames[idx]}',uppercoord,cv2.FONT_HERSHEY_SIMPLEX,1,(235,57,45),2)

        frame=findhands.drawLandmarks(frame, handlandmarks[0],False)
    cv2.imshow('Gesture Recognition',frame)
    if cv2.waitKey(5) & 0xff==ord('q'):
        run=False
cv2.destroyAllWindows()
camera.release()

if choice.lower()!='y':
    choice=input('Save the gesture data? (Y/n): ')
    if choice.lower()=='y':
        if os.path.exists('gesture_data.pkl'):
            print('Overwritten existing file')
        with open('gesture_data.pkl','wb') as f:
            pickle.dump(gesturenames,f)
            pickle.dump(knowngestures,f)
            print('Data saved succesfully!')