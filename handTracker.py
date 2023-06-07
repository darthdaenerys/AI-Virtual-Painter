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
     
    def drawLandmarks(self,frame,data,auto_draw=False):
        if auto_draw:
            if data.multi_hand_landmarks!=None:
                for hand in data.multi_hand_landmarks:
                    self.mpdraw.draw_landmarks(frame, hand,self.mp.solutions.hands.HAND_CONNECTIONS)
        else:
            allhands=data
            for myHand in allhands:
                cv2.line(frame,(myHand[0][0],myHand[0][1]),(myHand[1][0],myHand[1][1]),(255,0,255),2)
                cv2.line(frame,(myHand[1][0],myHand[1][1]),(myHand[2][0],myHand[2][1]),(255,0,255),2)
                cv2.line(frame,(myHand[2][0],myHand[2][1]),(myHand[3][0],myHand[3][1]),(255,0,255),2)
                cv2.line(frame,(myHand[3][0],myHand[3][1]),(myHand[4][0],myHand[4][1]),(255,0,255),2)
                cv2.line(frame,(myHand[0][0],myHand[0][1]),(myHand[5][0],myHand[5][1]),(255,0,255),2)
                cv2.line(frame,(myHand[5][0],myHand[5][1]),(myHand[6][0],myHand[6][1]),(255,0,255),2)
                cv2.line(frame,(myHand[6][0],myHand[6][1]),(myHand[7][0],myHand[7][1]),(255,0,255),2)
                cv2.line(frame,(myHand[7][0],myHand[7][1]),(myHand[8][0],myHand[8][1]),(255,0,255),2)
                cv2.line(frame,(myHand[0][0],myHand[0][1]),(myHand[17][0],myHand[17][1]),(255,0,255),2)
                cv2.line(frame,(myHand[17][0],myHand[17][1]),(myHand[18][0],myHand[18][1]),(255,0,255),2)
                cv2.line(frame,(myHand[18][0],myHand[18][1]),(myHand[19][0],myHand[19][1]),(255,0,255),2)
                cv2.line(frame,(myHand[19][0],myHand[19][1]),(myHand[20][0],myHand[20][1]),(255,0,255),2)
                cv2.line(frame,(myHand[5][0],myHand[5][1]),(myHand[9][0],myHand[9][1]),(255,0,255),2)
                cv2.line(frame,(myHand[9][0],myHand[9][1]),(myHand[13][0],myHand[13][1]),(255,0,255),2)
                cv2.line(frame,(myHand[13][0],myHand[13][1]),(myHand[17][0],myHand[17][1]),(255,0,255),2)
                cv2.line(frame,(myHand[9][0],myHand[9][1]),(myHand[10][0],myHand[10][1]),(255,0,255),2)
                cv2.line(frame,(myHand[10][0],myHand[10][1]),(myHand[11][0],myHand[11][1]),(255,0,255),2)
                cv2.line(frame,(myHand[11][0],myHand[11][1]),(myHand[12][0],myHand[12][1]),(255,0,255),2)
                cv2.line(frame,(myHand[13][0],myHand[13][1]),(myHand[14][0],myHand[14][1]),(255,0,255),2)
                cv2.line(frame,(myHand[14][0],myHand[14][1]),(myHand[15][0],myHand[15][1]),(255,0,255),2)
                cv2.line(frame,(myHand[15][0],myHand[15][1]),(myHand[16][0],myHand[16][1]),(255,0,255),2)
                for i in myHand:
                    cv2.circle(frame,(i[0],i[1]),4,(23,90,10),1)
                for i in myHand:
                    cv2.circle(frame,(i[0],i[1]),3,(255,255,125),-1)
        return frame

def findDistances(handData):
    distMatrix=np.zeros([len(handData),len(handData)],dtype=np.float32)
    palmSize=((handData[0][0]-handData[9][0])**2+(handData[0][1]-handData[9][1])**2)**.5
    for rows in range(0,len(handData)):
        for columns in range(0,len(handData)):
            distMatrix[rows][columns]=(((handData[rows][0]-handData[columns][0])**2+(handData[rows][1]-handData[columns][1])**2)**.5)/palmSize
    return distMatrix

def findError(knowngestures,unknownMatrix,keypoints):
    error=9999999
    idx=-1
    for i in range(len(knowngestures)):
        currenterror=0
        for rows in keypoints:
            for columns in keypoints:
                currenterror+=abs(knowngestures[i][rows][columns]-unknownMatrix[rows][columns])
        if currenterror<error:
            error=currenterror
            idx=i
    return error,idx