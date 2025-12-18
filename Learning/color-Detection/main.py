import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

vdo=cv.VideoCapture(0)
while True:
    _,frame=vdo.read()
    
    
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    
    red_lower1 = np.array([0, 120, 120])
    red_upper1 = np.array([5, 255, 255])

    red_lower2 = np.array([170, 120, 120])
    red_upper2 = np.array([180, 255, 255])

    mask1 = cv.inRange(hsv, red_lower1, red_upper1)
    mask2 = cv.inRange(hsv, red_lower2, red_upper2)
    redM = mask1 + mask2
    kernal=np.ones((5,5),'uint8')
    
    redM=cv.dilate(redM,kernal)
    red=cv.bitwise_and(frame,frame,mask=redM)
    
    
    green_lower = np.array([35, 60, 60])
    green_upper = np.array([90, 255, 255])

    greenM=cv.inRange(hsv,green_lower,green_upper)
    
    greenM=cv.dilate(greenM,kernal)
    green=cv.bitwise_and(frame,frame,mask=greenM)
    
    blue_lower = np.array([95, 120, 70])
    blue_upper = np.array([130, 255, 255])

    blueM=cv.inRange(hsv,blue_lower,blue_upper)
    
    blueM=cv.dilate(blueM,kernal)
    blue=cv.bitwise_and(frame,frame,mask=blueM)
    
    
    cont,hier=cv.findContours(redM,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cont2,_=cv.findContours(greenM,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cont3,_=cv.findContours(blueM,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
    for c in cont:
        area=cv.contourArea(c)
        if area>300:
            x,y,w,h = cv.boundingRect(c)
            frame=cv.rectangle(frame,(x,y),(x+w , y+h),(0,0,255),2)
            cv.putText(frame,"Red color",(x,y-20),cv.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,255))
            
    for c in cont2:
        area=cv.contourArea(c)
        if area>300:
            x,y,w,h = cv.boundingRect(c)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv.putText(frame,"Green color",(x,y-20),cv.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0))
            
    for c in cont3:
        area=cv.contourArea(c)
        if area>300:
            x,y,w,h = cv.boundingRect(c)
            cv.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            cv.putText(frame,"Blue color",(x,y-20),cv.FONT_HERSHEY_SCRIPT_COMPLEX,1,(255,0,0))
            
    
    cv.imshow("webcam",frame)
    k=cv.waitKey(3)
    if k==ord('q'):
        break
vdo.release()
cv.destroyAllWindows()