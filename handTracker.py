import cv2
import numpy as np

width=1280
height=720

class MediapipeHands:
    import mediapipe as mp
    def __init__(self,static_image_mode=False,max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands=self.mp.solutions.hands.Hands(static_image_mode,max_num_hands,model_complexity,min_detection_confidence,min_tracking_confidence)
        self.mpdraw=self.mp.solutions.drawing_utils
    
    def handsdata(self,frame,auto_draw=False):
        if auto_draw:
            return self.hands.process(frame)
        else:
            frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            allhands=[]
            handstype=[]
            results=self.hands.process(frame)
            if results.multi_hand_landmarks!=None:
                for hand in results.multi_handedness:
                    for handtype in hand.classification:
                        handstype.append(handtype.label)
                for hand in results.multi_hand_landmarks:
                    singlehand=[]
                    for landmark in hand.landmark:
                        singlehand.append((int(landmark.x*width),int(landmark.y*height)))
                    allhands.append(singlehand)
            return allhands,handstype